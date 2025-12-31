"""
LangChain Agent planner for Meetara Core with tool orchestration.
"""
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from langchain.tools import BaseTool

from app.agent.planner_utils import (
    build_page_filter,
    image_has_content,
    normalize_page_image_captions,
)
from app.core.config import settings
from app.core.logger import agent_logger, is_verbose
from app.core.query_analyzer import get_query_analyzer
from app.core.config_loader import config_loader  # ✅ For config-driven academic domains
from app.utils.image_path_utils import normalize_image_path_to_url  # ✅ Optimized image path handling
from app.services.image_descriptor import get_image_descriptor
from app.services.image_generator import get_image_generator
from app.core.llm_processor import get_llm_processor  # New import
from app.agent.tools import (
    AdapterSelectorTool,
    TranslationTool,
    SpeechTool,
    EmotionTool,
    FaceEmotionTool
)
from app.agent.mcp_router import get_mcp_router
from app.rag.domain_retrievers import get_domain_retriever


FOLLOW_UP_WORD_THRESHOLD = 14
FOLLOW_UP_CHAR_THRESHOLD = 120
FOLLOW_UP_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class DomainDetectionResult:
    """Aggregated result for domain detection and document retrieval."""

    domain: str
    intent: Optional[str]
    confidence: float
    analysis: Dict[str, Any]
    documents: List[Any]
    method: str


class MeetaraAgent:
    """Main agent class for Meetara Core with tool orchestration."""
    
    def __init__(self):
        self.llm = None # No LLM for offline operation
        self.tools = self._create_tools()
        # Memory is now handled by the agent's checkpointer
        self.memory = None
        self.agent_executor = self._create_agent_executor()
        self.query_analyzer = get_query_analyzer()
        self.llm_processor = get_llm_processor()  # Initialize smart LLM processor
        self.image_descriptor = get_image_descriptor()  # ✅ For enhanced image descriptions
        self.image_generator = get_image_generator()  # ✅ For RAG-based image generation
        # ✅ Conversation memory: session_id -> conversation history
        self.conversation_memory: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # ✅ Track the last confirmed domain per session for follow-up routing
        self.session_domain_memory: Dict[str, str] = {}
        # ✅ Cache academic domains from config (no hard-coding)
        self.academic_domains = set(config_loader.get_academic_domains())
        # ✅ Cache stop words from config (reusable, no hard-coding)
        self.stop_words = config_loader.get_stop_words()
        # ✅ Track caption assignment per page to distribute unique captions
        self._page_caption_trackers: Dict[tuple, Dict[str, Any]] = {}
        # ✅ Cache for locating PDF paths and page text extraction
        self._pdf_path_cache: Dict[str, Optional[Path]] = {}
        self._pdf_page_text_cache: Dict[tuple, str] = {}
        agent_logger.info("MeetaraAgent initialized with tools, memory, and config-driven query analyzer")
    
    def _get_conversation_history(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        if not session_id:
            return []
        history = self.conversation_memory.get(session_id, [])
        if history:
            agent_logger.info(f"Retrieved {len(history)} previous messages for session: {session_id}")
        return history
    
    def _store_conversation(self, session_id: Optional[str], query: str, response: str) -> None:
        if not session_id:
            return
        history = self.conversation_memory[session_id]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
        if len(history) > 20:
            self.conversation_memory[session_id] = history[-20:]
        agent_logger.info(f"Stored conversation in memory for session: {session_id}")
    
    def _merge_conversation_history(
        self,
        session_history: List[Dict[str, Any]],
        external_history: Optional[List[Any]]
    ) -> List[Dict[str, Any]]:
        """Merge stored session history with externally provided conversation history."""
        merged_history: List[Dict[str, Any]] = list(session_history) if session_history else []
        
        if external_history:
            normalized_history: List[Dict[str, Any]] = []
            for entry in external_history:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    normalized_history.append({
                        "role": entry["role"],
                        "content": entry["content"]
                    })
                elif isinstance(entry, str):
                    normalized_history.append({"role": "user", "content": entry})
            
            if normalized_history:
                merged_history.extend(normalized_history)
        
        # Keep the recent portion only to avoid unbounded growth
        if len(merged_history) > 40:
            merged_history = merged_history[-40:]
        
        return merged_history
    
    def _determine_preferred_domain(
        self,
        session_id: Optional[str],
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """
        Determine which domain should be preferred for follow-up routing.
        
        Priority order:
        1. Explicit domain from context (context.domain, context.conversation_domain, context.topic)
        2. Domain from conversation_history (extract from previous assistant responses)
        3. Domain from session memory
        """
        # Priority 1: Explicit domain from context
        if context:
            for key in ("domain", "conversation_domain", "topic"):
                domain_value = context.get(key)
                if domain_value:
                    agent_logger.debug(f"   Preferred domain from context.{key}: {domain_value}")
                    return domain_value
        
        # Priority 2: Extract domain from conversation_history
        if conversation_history:
            # Look for domain in recent assistant responses
            for msg in reversed(conversation_history[-10:]):  # Check last 10 messages
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    # Check if response metadata contains domain
                    content = msg.get("content", "")
                    metadata = msg.get("metadata", {})
                    if metadata.get("domain"):
                        domain = metadata.get("domain")
                        agent_logger.debug(f"   Preferred domain from conversation_history metadata: {domain}")
                        return domain
                    # Also check if domain is in the response itself (some formats include it)
                    if "domain" in msg:
                        domain = msg.get("domain")
                        agent_logger.debug(f"   Preferred domain from conversation_history: {domain}")
                        return domain
        
        # Priority 3: Session memory
        session_domain = self._get_session_domain(session_id)
        if session_domain:
            agent_logger.debug(f"   Preferred domain from session memory: {session_domain}")
            return session_domain
        
        return None
    
    def _get_session_domain(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        return self.session_domain_memory.get(session_id)
    
    def _update_session_domain(self, session_id: Optional[str], domain: str) -> None:
        if session_id and domain:
            self.session_domain_memory[session_id] = domain
    
    def _is_follow_up_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]],
        preferred_domain: Optional[str] = None
    ) -> bool:
        """
        Detect if query is a follow-up question.
        
        Can detect follow-ups even without conversation_history if preferred_domain is provided
        (indicating this is part of an ongoing conversation from external context).
        """
        query_lower = query.lower().strip()
        follow_up_phrases = [
            "what should i do",
            "what do i do",
            "what now",
            "what next",
            "next steps",
            "next step",
            "how can i",
            "how do i",
            "can you tell me more",
            "tell me more",
            "can you help me more",
            "can you help me again",
            "help me more",
            "i'm still",
            "i don't know",
            "any other advice",
            "anything else",
            "what else",
            "what else should",
            "should i call someone",
            "i'm worried",
            "i'm concerned",
            "how can i apply",
            "how do i apply",
            "can i apply",
            "should i consider",
            "what should i consider"
        ]
        
        # Check for explicit follow-up phrases (works even without history)
        if any(phrase in query_lower for phrase in follow_up_phrases):
            return True
        
        # If we have conversation history, use it for detection
        if conversation_history and len(conversation_history) > 0:
            # Short queries with history are likely follow-ups
            word_count = len(query_lower.split())
            if word_count <= FOLLOW_UP_WORD_THRESHOLD:
                return True
            
            if len(query_lower) <= FOLLOW_UP_CHAR_THRESHOLD:
                return True
        
        # If we have preferred_domain from context but no history, 
        # still consider short queries as follow-ups
        if preferred_domain and not conversation_history:
            word_count = len(query_lower.split())
            if word_count <= 8:  # Very short queries are likely follow-ups
                return True
        
        return False
    
    def _should_override_with_conversation_domain(
        self,
        query: str,
        detected_domain: str,
        preferred_domain: Optional[str],
        confidence: float,
        conversation_history: Optional[List[Dict[str, Any]]],
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if we should override detected domain with conversation domain.
        
        This function is domain-agnostic and works for ALL domains.
        
        Override conditions:
        1. preferred_domain exists and differs from detected_domain
        2. Query is a follow-up (short, context-dependent, or explicit follow-up phrase)
        3. Either:
           a) Confidence is low (< threshold), OR
           b) Domain was explicitly provided in context (force override for follow-ups)
        """
        if not preferred_domain or detected_domain == preferred_domain:
            return False
        
        # Check if this is a follow-up query (now works even without history if preferred_domain exists)
        is_follow_up = self._is_follow_up_query(query, conversation_history, preferred_domain)
        if not is_follow_up:
            agent_logger.debug(
                f"   Not a follow-up query: query='{query[:50]}...', "
                f"preferred_domain={preferred_domain}, history_len={len(conversation_history) if conversation_history else 0}"
            )
            return False
        
        # If domain was explicitly provided in context, ALWAYS override for follow-ups
        # This works for ANY domain (general_health, academic_tutoring, mental_health, etc.)
        domain_explicitly_provided = False
        if context:
            domain_explicitly_provided = any(
                context.get(key) == preferred_domain 
                for key in ("domain", "conversation_domain", "topic")
            )
        
        if domain_explicitly_provided:
            agent_logger.info(
                f"🔁 Follow-up override (explicit domain): query='{query[:50]}...', "
                f"detected={detected_domain} (conf={confidence:.3f}), "
                f"preferred={preferred_domain} (from context)"
            )
            return True
        
        # Only override if confidence is low (high confidence = clear domain change)
        if confidence >= FOLLOW_UP_CONFIDENCE_THRESHOLD:
            agent_logger.debug(
                f"   High confidence ({confidence:.3f}) prevents override: "
                f"detected={detected_domain}, preferred={preferred_domain}"
            )
            return False
        
        agent_logger.info(
            f"🔁 Follow-up override conditions met: query='{query[:50]}...', "
            f"detected={detected_domain} (conf={confidence:.3f}), preferred={preferred_domain}"
        )
        return True
    
    def _retrieve_documents_for_domain(
        self,
        query: str,
        domain: str,
        top_k: int = 3
    ) -> List[Any]:
        try:
            retriever = get_domain_retriever(domain)
            docs = retriever.similarity_search(query=query, k=top_k)
            agent_logger.info(
                f"🔁 Retrieved {len(docs)} documents directly from domain '{domain}' for follow-up query routing"
            )
            return docs
        except Exception as exc:
            agent_logger.warning(f"Could not retrieve documents for forced domain '{domain}': {exc}")
            return []
    
    async def _detect_domain_and_context(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        use_semantic_detection: bool,
        preferred_domain: Optional[str],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> DomainDetectionResult:
        context_domain = preferred_domain or (context.get("topic") if context else None)
        override_applied = False
        
        if use_semantic_detection:
            from app.rag.semantic_domain_detector import get_semantic_domain_detector
            
            semantic_detector = get_semantic_domain_detector()
            top_keyword_candidates = self.query_analyzer.get_top_candidate_domains(
                query=query,
                top_n=5,
                context_domain=context_domain
            )
            keyword_hints = [domain for domain, _ in top_keyword_candidates] if top_keyword_candidates else None
            keyword_analysis = self.query_analyzer.analyze_query(query=query, context_domain=context_domain)
            
            semantic_result = semantic_detector.hybrid_detect(
                query=query,
                keyword_hint_domains=keyword_hints
            )
            
            detected_domain = semantic_result["detected_domain"]
            detected_intent = keyword_analysis.get("detected_intent")
            confidence = semantic_result["confidence"]
            documents = semantic_result.get("documents", []) or []
            
            if self._should_override_with_conversation_domain(
                query,
                detected_domain,
                preferred_domain,
                confidence,
                conversation_history,
                context
            ):
                override_applied = True
                forced_domain = preferred_domain
                agent_logger.warning(
                    f"🔁 Follow-up detected. Overriding domain from '{detected_domain}' "
                    f"to conversation domain '{forced_domain}'."
                )
                detected_domain = forced_domain
                forced_docs = self._retrieve_documents_for_domain(query, forced_domain) if forced_domain else []
                if forced_docs:
                    documents = forced_docs
                confidence = max(confidence, 0.4)
            
            agent_logger.info(
                f"✅ Semantic-first detection - Domain: {detected_domain}, "
                f"Confidence: {confidence:.3f}, Docs: {len(documents)}"
            )
            
            analysis = {
                "detected_domain": detected_domain,
                "detected_intent": detected_intent,
                "confidence": confidence,
                "method": "semantic_first",
                "conversation_override": override_applied,
                "preferred_domain": preferred_domain
            }
            
            return DomainDetectionResult(
                domain=detected_domain,
                intent=detected_intent,
                confidence=confidence,
                analysis=analysis,
                documents=documents,
                method="semantic_first"
            )
        
        analysis = self.query_analyzer.analyze_query(query=query, context_domain=context_domain)
        detected_domain = analysis["detected_domain"]
        detected_intent = analysis["detected_intent"]
        confidence = analysis["confidence"]
        analysis["method"] = "keyword_first"
        
        agent_logger.info(f"Query analysis - Domain: {detected_domain}, Intent: {detected_intent}, Confidence: {confidence}")
        
        mcp_router = get_mcp_router()
        domain_context = await mcp_router._get_domain_context(query=query, domain=detected_domain)
        
        if self._should_override_with_conversation_domain(
            query,
            detected_domain,
            preferred_domain,
            confidence,
            conversation_history,
            context
        ):
            override_applied = True
            forced_domain = preferred_domain
            agent_logger.warning(
                f"🔁 Follow-up detected (keyword mode). Overriding domain from '{detected_domain}' "
                f"to conversation domain '{forced_domain}'."
            )
            detected_domain = forced_domain
            domain_context = self._retrieve_documents_for_domain(query, forced_domain) if forced_domain else domain_context
            confidence = max(confidence, 0.4)
        
        analysis["conversation_override"] = override_applied
        analysis["preferred_domain"] = preferred_domain
        
        return DomainDetectionResult(
            domain=detected_domain,
            intent=detected_intent,
            confidence=confidence,
            analysis=analysis,
            documents=domain_context or [],
            method="keyword_first"
        )
    
    async def _maybe_get_emotion_context(
        self,
        query: str,
        detected_domain: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not context or not context.get("emotion"):
            return None
        
        mcp_router = get_mcp_router()
        return await mcp_router.process_with_emotion_context(
            query=query,
            domain=detected_domain,
            detected_emotion=context["emotion"]
        )
    
    def _generate_images_from_context(
        self,
        query: str,
        domain_context: List[Any],
        detected_domain: str
    ) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        if not domain_context:
            return images
        
        should_gen, img_type, reasoning = self.image_generator.should_generate_image_from_rag(
            query=query,
            retrieved_docs=domain_context,
            domain=detected_domain,
            llm_processor=self.llm_processor
        )
        if should_gen and img_type:
            try:
                generated_image = self.image_generator.generate_from_query(query, detected_domain)
                if generated_image:
                    images.append(generated_image)
                    agent_logger.info(
                        f"🎨 Generated {img_type} image (reasoning: {reasoning}) for query: {query[:50]}"
                    )
            except Exception as exc:
                agent_logger.warning(f"Image generation failed (non-critical): {exc}")
        return images
    
    def _has_relevant_context(self, query: str, docs: List[Any]) -> bool:
        """
        Determine whether retrieved documents contain meaningful overlap with the query.
        Basic heuristic: look for shared keywords longer than 3 characters.
        """
        if not docs:
            return False
        
        import re
        query_terms = [
            term for term in re.findall(r'\w+', query.lower())
            if len(term) > 3
        ]
        # Special phrase check for quick short questions
        normalized_query = " ".join(query.lower().split())
        
        if not query_terms:
            return False
        
        for doc in docs[:5]:
            content = getattr(doc, "page_content", "") or ""
            content_lower = content.lower()
            if normalized_query in content_lower:
                return True
            
            matches = {term for term in query_terms if term in content_lower}
            if len(matches) >= 2:
                return True
        return False
    
    def _run_llm_pipeline(
        self,
        docs: List[Any],
        query: str,
        domain: str,
        intent: Optional[str],
        analysis: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        payload = self._process_documents_with_llm(
            docs=docs,
            original_query=query,
            domain=domain,
            intent=intent,
            analysis=analysis,
            conversation_history=conversation_history
        )
        
        if isinstance(payload, dict):
            response = payload.get("response", str(payload))
            images = payload.get("images", []) or []
        else:
            response = payload
            images = []
        return response, images
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "total_tools": len(self.tools),
            "tool_names": [tool.name for tool in self.tools],
            "tool_descriptions": [tool.description for tool in self.tools]
        }
    
    def get_memory(self) -> List[Any]:
        """Get memory information."""
        # Since we're using the new LangChain API, memory is handled differently
        # Return empty list for now as memory is managed by the agent's checkpointer
        return []
    
    def _create_tools(self) -> List[BaseTool]:
        """Create and configure all available tools."""
        tools = [
            AdapterSelectorTool(),
            TranslationTool(),
            SpeechTool(),
            EmotionTool(),
            FaceEmotionTool(),
        ]
        agent_logger.info(f"Created {len(tools)} tools for agent")
        return tools
    
    def _create_agent_executor(self):
        """Create the agent executor with tools and memory.
        
        Note: We intentionally use a mock executor because:
        1. Our Meetara GGUF models are custom format (not LangChain-compatible)
        2. Actual LLM processing happens in _process_documents_with_llm() when RAG finds documents
        3. This mock is only used as fallback when no documents are found
        """
        agent_logger.info("Initializing mock agent executor (RAG + GGUF models handle actual processing)")
        
        # Mock executor for fallback (when no documents found)
        class MockAgentExecutor:
            def __init__(self, tools, memory):
                self.tools = tools
                self.memory = memory
                self.verbose = True
            
            async def ainvoke(self, input_data):
                query = input_data.get("input", "")
                context = input_data.get("context", {})
                
                # Simple response generation based on query content
                if "health" in query.lower():
                    response = "I'm here to help with your health concerns. How can I assist you today?"
                elif "study" in query.lower():
                    response = "I can help you with study techniques and academic guidance."
                elif "stress" in query.lower():
                    response = "I understand stress can be challenging. Let me help you find ways to manage it."
                elif "time" in query.lower():
                    response = "Time management is crucial. I can help you develop effective strategies."
                else:
                    response = "I'm here to help you. How can I assist you today?"
                
                return {
                    "output": response,
                    "intermediate_steps": [],
                    "context": context
                }
        
        return MockAgentExecutor(self.tools, self.memory)
    
    async def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_semantic_detection: bool = True  # ✅ New: Enable semantic-first detection
    ) -> Dict[str, Any]:
        """
        Process a query through the Meetara agent system with conversation memory.
        
        Args:
            query: User query
            session_id: Optional session ID for conversation memory
            context: Optional context dict with topic, emotion, etc.
            use_semantic_detection: If True, use semantic-first domain detection (scalable)
                                   If False, use keyword-based detection (current method)
        """
        try:
            overall_start = time.time()
            stage_start = overall_start
            request_timestamp = datetime.now(timezone.utc).isoformat()
            session_history = self._get_conversation_history(session_id)
            external_history = context.get("conversation_history") if context else None
            conversation_history = self._merge_conversation_history(session_history, external_history)
            
            # Determine preferred domain (now also checks conversation_history)
            preferred_domain = self._determine_preferred_domain(session_id, context, conversation_history)
            
            # Debug logging for follow-up detection (always log, even if None)
            session_domain_from_memory = self._get_session_domain(session_id) if session_id else None
            agent_logger.info(
                f"🔍 Follow-up detection check: preferred_domain={preferred_domain}, "
                f"session_id={session_id}, "
                f"session_domain_from_memory={session_domain_from_memory}, "
                f"conversation_history_len={len(conversation_history)}, "
                f"context.domain={context.get('domain') if context else None}, "
                f"context.keys={list(context.keys()) if context else []}"
            )
            
            # Check if this is a follow-up query
            is_likely_follow_up = self._is_follow_up_query(query, conversation_history, preferred_domain or session_domain_from_memory)
            
            # Warn if follow-up detection is impaired due to missing context
            if is_likely_follow_up and not preferred_domain and not session_domain_from_memory:
                # Check if session_id is being sent but is different each time (common issue)
                session_id_provided = session_id is not None
                session_id_issue = ""
                if session_id_provided:
                    session_id_issue = f"\n     ⚠️ CRITICAL: session_id='{session_id}' is provided BUT session_domain_from_memory=None.\n     This means meetara-lab is sending a DIFFERENT session_id for each query!\n     meetara-lab MUST reuse the SAME session_id for all queries in a conversation."
                
                agent_logger.error(
                    f"🚨 CRITICAL FOLLOW-UP ROUTING FAILURE 🚨\n"
                    f"   Query: '{query[:100]}...'\n"
                    f"   Issue: Follow-up query detected but NO context available for domain routing!\n"
                    f"   Missing:\n"
                    f"     - session_id provided: {session_id_provided} (current: {session_id}){session_id_issue}\n"
                    f"     - context.domain: {context.get('domain') is None if context else True} (meetara-lab must send domain from Turn 1)\n"
                    f"     - conversation_history: {len(conversation_history) == 0} (meetara-lab must send conversation history)\n"
                    f"   Result: Query will be routed to semantically detected domain (may be WRONG domain)\n"
                    f"   Fix: meetara-lab MUST:\n"
                    f"     1. Generate ONE session_id per conversation (e.g., 'conv-abc123')\n"
                    f"     2. Reuse the SAME session_id for ALL queries in that conversation\n"
                    f"     3. OR send context.domain with domain from Turn 1\n"
                    f"     4. OR send context.conversation_history with domain metadata"
                )
            elif is_likely_follow_up:
                agent_logger.info(
                    f"✅ Follow-up query detected: '{query[:50]}...' "
                    f"(preferred_domain={preferred_domain}, session_memory={session_domain_from_memory})"
                )
            
            # If preferred_domain is None but we have session memory, use it as fallback for follow-ups
            # This handles cases where meetara-lab doesn't send context.domain or conversation_history
            if not preferred_domain and session_domain_from_memory:
                # Check if this looks like a follow-up query
                if is_likely_follow_up:
                    agent_logger.warning(
                        f"⚠️ Follow-up query detected but no preferred_domain from context. "
                        f"Using session memory domain '{session_domain_from_memory}' as fallback."
                    )
                    preferred_domain = session_domain_from_memory
                else:
                    agent_logger.debug(
                        f"   Session memory has domain '{session_domain_from_memory}' but query doesn't look like follow-up. "
                        f"Not overriding semantic detection."
                    )
            
            detection = await self._detect_domain_and_context(
                query,
                context,
                use_semantic_detection,
                preferred_domain,
                conversation_history
            )
            if is_verbose():
                agent_logger.info(f"⏱️ Domain detection & retrieval took {time.time() - stage_start:.2f}s")
            stage_start = time.time()
            emotion_context = await self._maybe_get_emotion_context(query, detection.domain, context)
            
            domain_context = detection.documents
            # For fine-tuning: optionally bypass relevance filtering to use ALL retrieved docs
            if settings.filter_rag_context_by_relevance:
                has_relevant_context = self._has_relevant_context(query, domain_context)
                docs_for_llm = domain_context if has_relevant_context else []
            else:
                # Fine-tuning mode: use all retrieved documents regardless of keyword matching
                has_relevant_context = bool(domain_context)
                docs_for_llm = domain_context
                if domain_context:
                    agent_logger.info(f"🔧 Fine-tuning mode: Using ALL {len(domain_context)} retrieved documents (relevance filtering disabled)")
            applied_domain = detection.domain if has_relevant_context else "general_knowledge"
            images = []
            if has_relevant_context:
                images = self._generate_images_from_context(query, domain_context, detection.domain)
            if is_verbose():
                agent_logger.info(f"⏱️ Image generation preparation took {time.time() - stage_start:.2f}s")
            stage_start = time.time()
            has_documents = bool(docs_for_llm)
            
            if has_documents:
                agent_logger.info(f"Processing {len(docs_for_llm)} documents with LLM for domain: {applied_domain}")
            else:
                agent_logger.info(f"No relevant documents found; using GGUF LLM general knowledge (domain: {applied_domain})")
            
            response, doc_images = self._run_llm_pipeline(
                docs=docs_for_llm,
                    query=query,
                domain=applied_domain,
                intent=detection.intent,
                analysis=detection.analysis,
                conversation_history=conversation_history
            )
            if is_verbose():
                agent_logger.info(f"⏱️ LLM generation took {time.time() - stage_start:.2f}s")
            
            if doc_images:
                images.extend(doc_images)
            
            if has_documents:
                agent_logger.info(f"✅ Generated response using Meetara GGUF + {len(docs_for_llm)} documents")
            else:
                agent_logger.info("✅ Generated response using Meetara GGUF general knowledge (no RAG)")
            
            if images:
                agent_logger.info(f"📷 Found {len(images)} associated images")
                for i, img in enumerate(images[:3]):
                    agent_logger.info(
                        f"   Image {i+1}: url={img.get('image_url', 'NO URL')}, path={img.get('image_path', 'NO PATH')}"
                    )
            else:
                agent_logger.debug("No images associated with this response")
            
            self._store_conversation(session_id, query, response)
            self._update_session_domain(session_id, applied_domain)
            response_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Determine RAG status based on whether documents were used
            docs_used = len(docs_for_llm) if docs_for_llm else 0
            if docs_used > 0 and detection.confidence >= 0.4:
                rag_status = "rag"  # RAG documents were used
            elif docs_used > 0:
                rag_status = "mixed"  # RAG + LLM knowledge
            else:
                rag_status = "llm"  # Pure LLM knowledge
            
            response_dict = {
                "response": response,
                "domain": applied_domain,
                "emotion_context": emotion_context,
                "tool_calls": [],
                "confidence": detection.confidence,
                "images": images,  # ✅ Include images if available (always initialized)
                "request_timestamp": request_timestamp,
                "response_timestamp": response_timestamp,
                "rag_status": rag_status,  # ✅ RAG status for UI
                "documents_used": docs_used,  # ✅ Number of documents used
            }
            agent_logger.info(f"✅ Total pipeline time {time.time() - overall_start:.2f}s")
            
            # ✅ Debug: Log what we're returning
            if images:
                agent_logger.info(f"✅ Returning response with {len(images)} images")
                agent_logger.debug(f"   Response keys: {list(response_dict.keys())}")
            else:
                agent_logger.debug(f"   Returning response with 0 images (images list is empty)")
            
            return response_dict
            
        except Exception as e:
            agent_logger.error(f"Error processing query: {e}")
            error_timestamp = datetime.now(timezone.utc).isoformat()
            return {
                "response": "I apologize, but I'm having trouble processing your request right now.",
                "domain": "general",
                "emotion_context": None,
                "tool_calls": [],
                "confidence": 0.0,
                "request_timestamp": request_timestamp if 'request_timestamp' in locals() else None,
                "response_timestamp": error_timestamp,
            }
    
    def _process_documents_config_driven(
        self, 
        docs: List, 
        original_query: str, 
        domain: str, 
        intent: Optional[str], 
        analysis: Dict[str, Any]
    ) -> str:
        """Process documents using config-driven content filtering and response templates."""
        # Get response template based on domain and intent
        template = self.query_analyzer.get_response_template(domain, intent)
        
        # Collect all relevant information from documents
        all_relevant_info = set()
        
        # Process each document using config-driven filtering
        for doc in docs[:3]:  # Look at top 3 docs
            content = doc.page_content.strip()
            if content:
                # Use config-driven content filtering
                relevant_sentences = self.query_analyzer.filter_relevant_content(
                    content, original_query.lower(), domain, intent
                )
                all_relevant_info.update(relevant_sentences)
        
        # Build response using config-driven template
        # ⚠️ WARNING: This is a FALLBACK method. Should only be used if LLM fails.
        agent_logger.warning(f"⚠️ Using config-driven fallback (LLM failed) for domain: {domain}")
        
        response_parts = []
        
        # Add template-based introduction (only if not empty)
        if template.get("intro"):
            response_parts.append(template["intro"])
        
        # Add template-based disclaimer (only if not empty - medical domains only)
        if template.get("disclaimer"):
            response_parts.append(template["disclaimer"])
        
        # Add template-based header (only if not empty)
        if template.get("header"):
            response_parts.append(template["header"])
        
        # Add filtered content
        if all_relevant_info:
            # Sort for consistent output and format as bullet points
            sorted_info = sorted(list(all_relevant_info))
            # Format as bullet points instead of raw text
            formatted_content = '\n'.join([f"• {info}" for info in sorted_info[:10]])  # Limit to top 10
            response_parts.append(formatted_content)
        else:
            response_parts.append("No specific information found for this query.")
        
        return '\n\n'.join(response_parts)
    
    def _process_documents_with_llm(
        self, 
        docs: List, 
        original_query: str, 
        domain: str, 
        intent: Optional[str], 
        analysis: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Process documents using smart LLM for truly intelligent responses."""
        try:
            # Reset per-call caption trackers so each run distributes captions cleanly
            self._page_caption_trackers = {}
            
            # Log image extraction status (useful for fine-tuning mode)
            if not settings.enable_image_extraction_during_query:
                agent_logger.info("⏭️ Image extraction disabled during query (fine-tuning mode) - skipping image processing")

            # Extract document content AND metadata for source attribution and images
            context_docs = []
            document_metadata = []  # Store metadata for source citations
            associated_images = []  # Store images from documents
            relevant_pages = set()  # Track relevant pages to find adjacent images
            document_filenames = set()  # Track filenames to find related images
            # ✅ MULTI-PDF: Track which pages belong to which PDF
            pages_by_filename = {}  # {filename_base: set of page numbers}
            
            for doc in docs[:3]:  # Use top 3 most relevant documents
                content = doc.page_content.strip()
                if content:
                    context_docs.append(content)
                    
                    # Extract metadata for source attribution
                    metadata = doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {}
                    
                    # ✅ VALIDATION: Only include source if document actually has relevant content
                    # Check if document has substantial content OR has images related to the query
                    is_relevant_source = False
                    
                    # Check 1: Document has associated images (visual content is always relevant)
                    if 'associated_images' in metadata:
                        is_relevant_source = True
                    
                    # Check 2: Document has substantial content (not just a few words)
                    elif len(content) > 50:
                        # Extract meaningful query keywords (longer words, exclude stop words from config)
                        query_words = original_query.lower().split()
                        query_keywords = [word for word in query_words if len(word) > 3 and word not in self.stop_words]
                        
                        # Also extract key phrases (2-word combinations) for better matching
                        query_phrases = []
                        for i in range(len(query_words) - 1):
                            phrase = f"{query_words[i]} {query_words[i+1]}"
                            if len(phrase) > 6 and phrase not in self.stop_words:
                                query_phrases.append(phrase)
                        
                        content_lower = content.lower()
                        
                        # Check for keyword matches (at least 2 keywords OR 1 phrase)
                        keyword_matches = sum(1 for keyword in query_keywords if keyword in content_lower)
                        phrase_matches = sum(1 for phrase in query_phrases if phrase in content_lower)
                        
                        # Require meaningful keyword overlap to avoid false citations
                        # This ensures we only cite sources that actually discuss the topic
                        # Same criteria for all domains: at least 2 keyword matches OR 1 phrase match
                        is_relevant_source = keyword_matches >= 2 or phrase_matches > 0
                    
                    # Build source_info for tracking (always needed for adjacent image search)
                    source_info = {
                        'filename': metadata.get('file_name', metadata.get('source', 'Unknown document')),
                        'author': metadata.get('author', None),
                        'page': metadata.get('page', None),
                        'domain': domain
                    }
                    
                    # ✅ Always add to document_metadata if RAG retrieved it - semantic search already filtered for relevance
                    # The keyword matching is too strict and can exclude valid sources that semantic search found
                    document_metadata.append(source_info)
                    if not is_relevant_source:
                        # Log if keyword matching didn't pass, but still include for citation
                        filename = source_info.get('filename', 'Unknown document')
                        page = source_info.get('page', 'unknown')
                        agent_logger.debug(f"   Including source for citation (semantic match): {filename} page {page} (keyword match threshold not met, but semantically relevant)")
                    
                    # Track relevant pages and filenames for finding adjacent images (always track, even if not cited)
                    if source_info.get('page') is not None:
                        relevant_pages.add(source_info['page'])
                    if source_info.get('filename'):
                        # Extract base filename without extension for matching
                        filename_base = Path(source_info['filename']).stem
                        document_filenames.add(filename_base)
                        # ✅ MULTI-PDF: Track which pages belong to which PDF
                        if filename_base not in pages_by_filename:
                            pages_by_filename[filename_base] = set()
                        if source_info.get('page') is not None:
                            pages_by_filename[filename_base].add(source_info['page'])
                    
                    # ✅ Extract associated images from metadata (if enabled)
                    # Skip image extraction if disabled (useful for fine-tuning)
                    if not settings.enable_image_extraction_during_query:
                        agent_logger.debug(f"⏭️ Image extraction disabled during query (fine-tuning mode)")
                    else:
                        agent_logger.debug(f"Checking metadata for document from {source_info.get('filename')} (page {source_info.get('page')})")
                        agent_logger.debug(f"Metadata keys: {list(metadata.keys())}")
                    
                    if settings.enable_image_extraction_during_query and 'associated_images' in metadata:
                        import json
                        try:
                            # associated_images is stored as JSON string
                            raw_images = metadata['associated_images']
                            agent_logger.info(f"📷 Found 'associated_images' in metadata (type: {type(raw_images).__name__}, length: {len(str(raw_images))})")
                            images_data = json.loads(raw_images) if isinstance(raw_images, str) else raw_images
                            agent_logger.info(f"📷 Parsed associated_images: {len(images_data) if isinstance(images_data, list) else 'not a list'} images")
                            if isinstance(images_data, list):
                                for img in images_data:
                                    # Only add unique images (avoid duplicates)
                                    # ✅ OPTIMIZED: Use centralized image path normalization
                                    image_path = img.get('image_path', '')
                                    image_url, img_path = normalize_image_path_to_url(image_path, domain)
                                    if image_url:
                                        agent_logger.info(f"✅ Generated image URL: {image_url} from path: {image_path}")
                                    
                                    # Only add images that have valid URLs
                                    if image_url:
                                        # ✅ Extract figure_number from caption if missing
                                        figure_number = img.get('figure_number')
                                        if not figure_number:
                                            # Try to extract from caption
                                            caption = img.get('caption', '')
                                            if caption:
                                                import re
                                                match = re.search(r'(?:FIGURE|Figure)\s+([\d\.]+)', caption, re.IGNORECASE)
                                                if match:
                                                    figure_number = match.group(1)
                                            # If still no figure_number, try from captions array
                                            if not figure_number:
                                                captions = img.get('captions', [])
                                                for cap in captions:
                                                    if isinstance(cap, str):
                                                        match = re.search(r'(?:FIGURE|Figure)\s+([\d\.]+)', cap, re.IGNORECASE)
                                                        if match:
                                                            figure_number = match.group(1)
                                                            break
                                        
                                        img_info = {
                                            'image_path': image_path,  # Original path
                                            'image_url': image_url,  # ✅ API-accessible URL
                                            'image_id': img.get('image_id'),
                                            'page': img.get('page', source_info.get('page')),
                                            'filename': source_info.get('filename'),
                                            'ocr_text': img.get('ocr_text', '')[:200],  # Limit OCR text length
                                            'captions': img.get('captions', []),  # ✅ FIGURE captions from page (raw strings)
                                            'caption_details': img.get('caption_details', []),  # ✅ Structured caption objects
                                            'figure_number': figure_number,  # ✅ Extracted figure number (with fallback parsing)
                                            'caption': img.get('caption'),  # ✅ Primary caption text
                                            'image_size': img.get('image_size', [0, 0]),  # ✅ Image dimensions
                                            'format': img.get('format', 'PNG'),  # ✅ Image format
                                            'visual_description': self._get_enhanced_image_description(img),  # ✅ Enhanced description
                                            'page_source': 'metadata'
                                        }
                                        
                                        # ✅ FILTER: Check if image is relevant to the query
                                        # Use stricter relevance checking - don't include images just because they're in a relevant document
                                        is_relevant = self._is_image_relevant_to_query(img_info, original_query, domain)
                                        
                                        # Only include if actually relevant - don't auto-include for academic domains
                                        if not is_relevant:
                                            agent_logger.debug(f"   ⏭️ Skipped image: insufficient relevance to query '{original_query[:50]}...'")
                                        
                                        if is_relevant:
                                            # Check if this image was already added
                                            if not any(existing.get('image_path') == img_info['image_path'] 
                                                     for existing in associated_images):
                                                associated_images.append(img_info)
                                                agent_logger.info(f"   ✅ Added relevant image: {image_url}")
                                            else:
                                                agent_logger.debug(f"   Skipped duplicate image: {img_info.get('image_path', 'NO PATH')}")
                                        else:
                                            agent_logger.debug(f"   ⏭️ Skipped irrelevant image (doesn't match query): {image_url}")
                                    else:
                                        agent_logger.warning(f"   ⚠️ Skipping image with no valid URL: {image_path}")
                        except (json.JSONDecodeError, TypeError) as e:
                            agent_logger.warning(f"Failed to parse associated_images metadata: {e}")
                            agent_logger.warning(f"   Raw value: {raw_images[:200] if raw_images else 'None'}")
                    else:
                        agent_logger.info(f"❌ No 'associated_images' key in metadata for document from {source_info.get('filename')} (page {source_info.get('page')})")
                        agent_logger.debug(f"   Available metadata keys: {list(metadata.keys())}")
            
            # ✅ ENHANCEMENT: Find images on adjacent pages for ALL domains (LIMITED)
            # Only search adjacent pages if we have fewer than 3 relevant images (to avoid too many results)
            # This helps include related figures that appear on nearby pages in any document type
            # Skip if image extraction is disabled (fine-tuning mode)
            if settings.enable_image_extraction_during_query and relevant_pages and document_filenames and len(associated_images) < 3:
                agent_logger.info(f"🔍 Looking for images on adjacent pages (±2) for relevant pages: {sorted(relevant_pages)} (current images: {len(associated_images)})")
                # ✅ MULTI-PDF: Log which pages belong to which PDF
                if pages_by_filename:
                    for filename_base, pages in pages_by_filename.items():
                        agent_logger.info(f"   PDF {filename_base}: pages {sorted(pages)}")
                adjacent_images = self._find_adjacent_page_images(
                    relevant_pages=relevant_pages,
                    document_filenames=document_filenames,
                    pages_by_filename=pages_by_filename,  # ✅ MULTI-PDF: Pass page-to-PDF mapping
                    domain=domain,
                    original_query=original_query,
                    existing_images=associated_images
                )
                if adjacent_images:
                    # ✅ RAG-BASED: Rank adjacent images by semantic relevance + page proximity
                    # Images from same document are semantically relevant (RAG already determined this)
                    # But we should prioritize images with captions/OCR that match the query
                    scored_images = []
                    query_lower = original_query.lower()
                    
                    # Extract meaningful query terms for semantic matching
                    # Stop words are loaded from config (domain_keywords.yaml) - reusable across codebase
                    # These words appear in almost every sentence but don't help match content semantically
                    # Example: "How do you find the derivative?" → keywords: ["find", "derivative"] (not "how", "do", "you", "the")
                    query_words = query_lower.split()
                    query_terms = [word for word in query_words if len(word) > 2 and word not in self.stop_words]
                    
                    for img in adjacent_images:
                        img_page_raw = img.get('page', 0)
                        img_page = img_page_raw if isinstance(img_page_raw, (int, float)) else 0
                        score = 0
                        
                        # ✅ SEMANTIC RELEVANCE: Check if image content matches query
                        # This helps prioritize images with relevant captions/OCR even if further away
                        caption_text = (img.get('caption', '') or '').lower()
                        captions_text = ' '.join([str(c) for c in img.get('captions', [])]).lower()
                        ocr_text = (img.get('ocr_text', '') or '').lower()
                        figure_num = (img.get('figure_number', '') or '').lower()
                        
                        combined_image_text = f"{caption_text} {captions_text} {ocr_text} {figure_num}"
                        
                        # Score based on semantic matches (captions/OCR/figures matching query)
                        semantic_score = 0
                        if query_terms:
                            for term in query_terms:
                                if term in combined_image_text:
                                    # Caption matches are worth more (most reliable)
                                    if term in caption_text or term in captions_text:
                                        semantic_score += 5
                                    elif term in ocr_text:
                                        semantic_score += 2
                                    elif term in figure_num:
                                        semantic_score += 1
                        
                        # ✅ PAGE PROXIMITY: Boost score for images closer to relevant pages
                        proximity_score = 0
                        if relevant_pages:
                            try:
                                min_page_distance = min(abs(img_page - p) for p in relevant_pages if isinstance(p, (int, float)))
                            except ValueError:
                                min_page_distance = None
                            
                            if min_page_distance is None:
                                proximity_score = 1
                            else:
                                # Closer pages = higher score (but less important than semantic relevance)
                                if min_page_distance == 0:  # Same page (shouldn't happen, but include)
                                    proximity_score = 10
                                elif min_page_distance == 1:  # Adjacent page
                                    proximity_score = 5
                                elif min_page_distance == 2:  # 2 pages away
                                    proximity_score = 2
                                else:  # Further away
                                    proximity_score = 1
                        else:
                            proximity_score = 1  # Default if no relevant pages
                        
                        # ✅ COMBINED SCORE: Semantic relevance (more important) + proximity (tie-breaker)
                        # Semantic relevance is weighted higher because it indicates actual content match
                        total_score = (semantic_score * 2) + proximity_score
                        
                        scored_images.append((total_score, img_page, img))
                        
                        # Log scoring for debugging
                        if semantic_score > 0:
                            agent_logger.debug(f"   Image page {img_page}: semantic_score={semantic_score}, proximity_score={proximity_score}, total={total_score}")
                    
                    # Sort by total score (highest first), then by page proximity as tie-breaker
                    scored_images.sort(key=lambda x: (x[0], -x[1] if isinstance(x[1], (int, float)) else 0), reverse=True)
                    
                    # Take top 5 most relevant images (by semantic + proximity score)
                    limited_adjacent = [img for score, page, img in scored_images[:5]]
                    agent_logger.info(f"✅ Found {len(adjacent_images)} adjacent images, ranked by semantic relevance + proximity, selected top {len(limited_adjacent)} most relevant")
                    if limited_adjacent:
                        top_scores = [score for score, _, _ in scored_images[:len(limited_adjacent)]]
                        agent_logger.info(f"   Top image scores: {top_scores}")
                    associated_images.extend(limited_adjacent)
                else:
                    agent_logger.info(f"⚠️ No adjacent page images found for pages {sorted(relevant_pages)}")
            elif len(associated_images) >= 3:
                agent_logger.info(f"⏭️ Skipping adjacent page search: already have {len(associated_images)} relevant images")
            
            if associated_images:
                associated_images = normalize_page_image_captions(associated_images)
            
            # ✅ Generate intelligent response using local LLM with conversation history and source metadata
            # Note: associated_images are extracted separately and returned in response
            response = self.llm_processor.generate_intelligent_response(
                query=original_query,
                context_docs=context_docs,
                domain=domain,
                conversation_history=conversation_history,  # ✅ Pass conversation history
                document_metadata=document_metadata,  # ✅ Pass metadata for source citations
                associated_images=None  # ❌ Don't send image metadata to LLM; only use for UI display
            )
            
            # Return response with images (if extraction is enabled)
            agent_logger.info(f"📊 Summary: Processed {len(context_docs)} documents, found {len(associated_images)} images")
            if settings.enable_image_extraction_during_query and associated_images:
                # Limit total images to top 8 most relevant (to avoid overwhelming the UI)
                max_images = 8
                if len(associated_images) > max_images:
                    agent_logger.info(f"⚠️ Limiting images from {len(associated_images)} to top {max_images} most relevant")
                    associated_images = associated_images[:max_images]
                
                agent_logger.info(f"✅ Returning {len(associated_images)} images with response")
                for i, img in enumerate(associated_images):
                    agent_logger.info(f"   Image {i+1}: {img.get('image_url', 'NO URL')}")
                # Return dict with response and images
                return {
                    "response": response,
                    "images": associated_images
                }
            else:
                agent_logger.warning(f"⚠️ No images to return after processing {len(context_docs)} documents - returning string response only")
            
            return response
            
        except Exception as e:
            agent_logger.error(f"Error in LLM document processing: {e}")
            # ✅ CRITICAL: Preserve images even if LLM fails (if extraction is enabled)
            # If we have images, return them with fallback response
            if settings.enable_image_extraction_during_query and associated_images:
                agent_logger.warning(f"⚠️ LLM failed but preserving {len(associated_images)} images with fallback response")
                fallback_response = self._process_documents_config_driven(docs, original_query, domain, intent, analysis)
                return {
                    "response": fallback_response,
                    "images": associated_images[:8]  # Limit to top 8
                }
            # Fallback to config-driven processing (no images)
            return self._process_documents_config_driven(docs, original_query, domain, intent, analysis)
    
    def _is_image_relevant_to_query(self, img_info: Dict[str, Any], query: str, domain: str = None) -> bool:
        """
        Check if an extracted image is relevant to the user's query.
        
        ✅ RAG-BASED APPROACH: Trust semantic search results
        If an image came from a document retrieved by RAG semantic search, it's semantically relevant.
        Only filter out truly empty or invalid images.
        
        Args:
            img_info: Image metadata with ocr_text, visual_description, caption, etc.
            query: User's original query (for logging only - not used for filtering)
            domain: Domain context (for logging only)
            
        Returns:
            True if image has content and should be included, False otherwise
        """
        # ✅ RAG principle: If the document was semantically retrieved, its images are relevant
        # Only filter out images with no content at all
        
        if not image_has_content(img_info):
            agent_logger.debug(f"   Image filtered: no content (empty image)")
            return False
        
        # ✅ Trust RAG: Include all images with content from semantically retrieved documents
        return True
    
    def _find_adjacent_page_images(
        self,
        relevant_pages: set,
        document_filenames: set,
        pages_by_filename: Dict[str, set],  # ✅ MULTI-PDF: Map of filename_base -> set of pages
        domain: str,
        original_query: str,
        existing_images: List[Dict[str, Any]],
        page_range: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Find images on adjacent pages (±page_range) for relevant documents.
        
        This helps include related figures that appear on nearby pages in academic documents.
        For example, if page 1270 is relevant, also check pages 1268-1272 for related images.
        
        ✅ MULTI-PDF SUPPORT: Only searches adjacent pages within the same PDF.
        Prevents mixing page numbers across different PDFs (e.g., page 100 in PDF1 vs page 100 in PDF2).
        
        Args:
            relevant_pages: Set of page numbers that were found relevant (from all PDFs)
            document_filenames: Set of document filename stems (without extension)
            pages_by_filename: Dict mapping filename_base -> set of page numbers for that PDF
            domain: Domain name for image directory lookup
            original_query: Original user query for relevance checking
            existing_images: Already found images (to avoid duplicates)
            page_range: Number of pages to check on each side (default: 2)
            
        Returns:
            List of image metadata dictionaries for adjacent page images
        """
        adjacent_images = []
        existing_paths = {
            img.get('image_path', '').replace('\\', '/')
            for img in existing_images
            if img.get('image_path')
        }
        
        try:
            # ✅ MULTI-PDF: Build adjacent pages PER PDF (not across all PDFs)
            # This prevents searching PDF1 for pages that belong to PDF2
            adjacent_pages_by_filename = {}  # {filename_base: set of adjacent pages}
            
            if pages_by_filename:
                # Use the page-to-PDF mapping to build PDF-specific adjacent pages
                for filename_base, pdf_pages in pages_by_filename.items():
                    adjacent_pages_for_pdf = set()
                    for page in pdf_pages:
                        for offset in range(-page_range, page_range + 1):
                            if offset != 0:  # Skip the page itself (already processed)
                                adjacent_pages_for_pdf.add(page + offset)
                    if adjacent_pages_for_pdf:
                        adjacent_pages_by_filename[filename_base] = adjacent_pages_for_pdf
                        agent_logger.info(f"   PDF {filename_base}: checking {len(adjacent_pages_for_pdf)} adjacent pages: {sorted(adjacent_pages_for_pdf)}")
            else:
                # Fallback: If no mapping provided, use old logic (but less accurate)
                adjacent_pages = set()
                for page in relevant_pages:
                    for offset in range(-page_range, page_range + 1):
                        if offset != 0:
                            adjacent_pages.add(page + offset)
                # Assign same adjacent pages to all PDFs (not ideal but backwards compatible)
                for filename_base in document_filenames:
                    adjacent_pages_by_filename[filename_base] = adjacent_pages
                agent_logger.warning(f"   No page-to-PDF mapping provided, using fallback (may mix pages across PDFs)")
            
            if not adjacent_pages_by_filename:
                return []
            
            # ✅ STEP 1: Try to get rich metadata (captions, figure numbers) from vector store
            # Query vector store for documents on adjacent pages to get their associated_images metadata
            adjacent_images_from_store = []
            retriever = None
            try:
                from app.rag.domain_retrievers import get_domain_retriever
                retriever = get_domain_retriever(domain)
                
                # ✅ MULTI-PDF: For each PDF, search only its own adjacent pages
                for filename_base in document_filenames:
                    full_filename = f"{filename_base}.pdf"
                    # Get adjacent pages for THIS PDF only
                    adjacent_pages = adjacent_pages_by_filename.get(filename_base, set())
                    if not adjacent_pages:
                        continue
                    
                    for page_num in adjacent_pages:
                        try:
                            # Query vector store with filter for this specific page and filename
                            # Use a broad query to get the document, then filter by metadata
                            filter_dict = {
                                'file_name': full_filename,
                                'page': page_num
                            }
                            
                            # Get documents matching this page and filename using direct vectorstore query
                            # This is more efficient than similarity_search for exact metadata matches
                            try:
                                if hasattr(retriever, 'vectorstore') and retriever.vectorstore:
                                    # Use ChromaDB's get method with where filter for exact match
                                    results = retriever.vectorstore.get(
                                        where=build_page_filter(full_filename, page_num),
                                        limit=5  # Get a few chunks from this page
                                    )
                                    
                                    # Convert results to documents with metadata
                                    ids = results.get('ids', [])
                                    metadatas = results.get('metadatas', [])
                                    documents = results.get('documents', [])
                                    
                                    # Process each result
                                    for i, metadata in enumerate(metadatas):
                                        if metadata and 'associated_images' in metadata:
                                            # Parse associated_images metadata
                                            import json
                                            raw_images = metadata['associated_images']
                                            images_data = json.loads(raw_images) if isinstance(raw_images, str) else raw_images
                                            
                                            if isinstance(images_data, list):
                                                for img in images_data:
                                                    # Check if this image matches the page
                                                    img_page = img.get('page', page_num)
                                                    if img_page == page_num:
                                                        # Extract image path and construct image info
                                                        image_path = img.get('image_path', '')
                                                        if image_path:
                                                            # Normalize path for comparison
                                                            normalized_path = image_path.replace('\\', '/')
                                                            if normalized_path not in existing_paths:
                                                                # Build image URL
                                                                # ✅ OPTIMIZED: Use centralized image path normalization
                                                                image_url, _ = normalize_image_path_to_url(image_path, domain)
                                                                
                                                                # Create rich image metadata with captions and figure numbers
                                                                img_info = {
                                                                    'image_path': normalized_path,
                                                                    'image_url': image_url,
                                                                    'image_id': img.get('image_id', ''),
                                                                    'page': img_page,
                                                                    'filename': full_filename,
                                                                    'ocr_text': img.get('ocr_text', '')[:200],
                                                                    'captions': img.get('captions', []),  # ✅ Full captions from metadata
                                                                    'caption_details': img.get('caption_details', []),  # ✅ Structured caption data
                                                                    'figure_number': img.get('figure_number'),  # ✅ Figure number from metadata
                                                                    'caption': img.get('caption'),  # ✅ Primary caption text
                                                                    'image_size': img.get('image_size', [0, 0]),
                                                                    'format': img.get('format', 'PNG'),
                                                                    'visual_description': self._get_enhanced_image_description(img),
                                                                    'page_source': 'metadata'
                                                                }
                                                                
                                                                # ✅ RAG-BASED: Trust semantic relevance - if from same document, include it
                                                                # Images from the same document are semantically related (RAG determined this)
                                                                # Only require that image has content
                                                                if image_has_content(img_info):
                                                                    adjacent_images_from_store.append(img_info)
                                                                    agent_logger.info(f"   ✅ Found adjacent page image with metadata: {image_url} (page {page_num}, figure: {img_info.get('figure_number', 'N/A')})")
                                                                    break  # Found image, move to next result
                                                break  # Found associated_images, move to next page
                                else:
                                    # Fallback: use similarity_search if direct get not available
                                    docs = retriever.similarity_search(
                                        query="document page text content",
                                        k=10,
                                        filter_dict=filter_dict,
                                        use_cache=False
                                    )
                                    
                                    for doc in docs:
                                        metadata = doc.metadata if hasattr(doc, 'metadata') and doc.metadata else {}
                                        if (metadata.get('file_name') == full_filename and 
                                            metadata.get('page') == page_num and
                                            'associated_images' in metadata):
                                            
                                            import json
                                            raw_images = metadata['associated_images']
                                            images_data = json.loads(raw_images) if isinstance(raw_images, str) else raw_images
                                            
                                            if isinstance(images_data, list):
                                                for img in images_data:
                                                    img_page = img.get('page', page_num)
                                                    if img_page == page_num:
                                                        # ✅ OPTIMIZED: Use centralized image path normalization
                                                        image_path = img.get('image_path', '')
                                                        if image_path:
                                                            normalized_path = image_path.replace('\\', '/')
                                                            if normalized_path not in existing_paths:
                                                                image_url, _ = normalize_image_path_to_url(image_path, domain)
                                                                
                                                                if image_url:
                                                                    img_info = {
                                                                        'image_path': normalized_path,
                                                                        'image_url': image_url,
                                                                    'image_id': img.get('image_id', ''),
                                                                    'page': img_page,
                                                                    'filename': full_filename,
                                                                    'ocr_text': img.get('ocr_text', '')[:200],
                                                                    'captions': img.get('captions', []),
                                                                    'caption_details': img.get('caption_details', []),
                                                                    'figure_number': img.get('figure_number'),
                                                                    'caption': img.get('caption'),
                                                                    'image_size': img.get('image_size', [0, 0]),
                                                                    'format': img.get('format', 'PNG'),
                                                                    'visual_description': self._get_enhanced_image_description(img),
                                                                    'page_source': 'metadata'
                                                                }
                                                                
                                                                # ✅ RAG-BASED: Trust semantic relevance - if from same document, include it
                                                                # Images from the same document are semantically related (RAG determined this)
                                                                # Only require that image has content
                                                                if image_has_content(img_info):
                                                                    adjacent_images_from_store.append(img_info)
                                                                    agent_logger.info(f"   ✅ Found adjacent page image with metadata: {image_url} (page {page_num}, figure: {img_info.get('figure_number', 'N/A')})")
                                                                break
                                                break
                            except Exception as e:
                                agent_logger.debug(f"   Could not query vectorstore.get for page {page_num} of {full_filename}: {e}")
                        except Exception as e:
                            agent_logger.debug(f"   Could not get metadata for page {page_num} of {full_filename}: {e}")
                            continue
            except Exception as e:
                agent_logger.debug(f"   Could not query vector store for adjacent page metadata: {e}")
            
            # ✅ STEP 2: Also look in images directory for any images we might have missed
            # (This is a fallback for images that don't have vector store metadata)
            # Images are stored at project root: images/{domain}/{filename}/
            # Try both relative to vectorstore parent and project root
            project_root = settings.vectorstore_path.parent
            images_dir = project_root / "images" / domain
            if not images_dir.exists():
                # Fallback: try current working directory
                images_dir = Path.cwd() / "images" / domain
            if not images_dir.exists():
                agent_logger.debug(f"   Images directory not found: {images_dir}")
                return []
            
            # ✅ MULTI-PDF: For each document filename, search for images on adjacent pages
            # Each PDF only searches its own adjacent pages (from adjacent_pages_by_filename)
            for filename_base in document_filenames:
                # Get adjacent pages for THIS PDF only
                adjacent_pages = adjacent_pages_by_filename.get(filename_base, set())
                if not adjacent_pages:
                    agent_logger.debug(f"   No adjacent pages for PDF {filename_base}, skipping")
                    continue
                
                # Find directory matching this filename
                doc_image_dir = images_dir / filename_base
                if not doc_image_dir.exists():
                    agent_logger.debug(f"   Image directory not found for {filename_base}: {doc_image_dir}")
                    continue
                
                # Scan for image files matching adjacent page numbers for THIS PDF
                for image_file in doc_image_dir.glob("*.png"):
                    try:
                        # Extract page number from filename: "filename_page_1271_img__Im0.png"
                        filename_parts = image_file.stem.split('_page_')
                        if len(filename_parts) < 2:
                            continue
                        
                        page_part = filename_parts[1].split('_img__')[0]
                        page_num = int(page_part)
                        
                        # ✅ MULTI-PDF: Check if this page is in THIS PDF's adjacent pages list
                        if page_num not in adjacent_pages:
                            continue
                        
                        # Normalize path to match stored format (relative to images directory)
                        # Stored format: "images/academic_tutoring/filename/filename_page_1270_img__Im0.png"
                        image_path_str = f"images/{domain}/{filename_base}/{image_file.name}".replace('\\', '/')
                        
                        # Skip if already found from vector store or in existing images
                        if image_path_str in existing_paths:
                            continue
                        
                        # Check if we already have this image from vector store metadata
                        already_found = any(
                            img.get('image_path', '').replace('\\', '/') == image_path_str 
                            for img in adjacent_images_from_store
                        )
                        if already_found:
                            continue
                        
                        # ✅ OPTIMIZED: Use centralized image path normalization
                        image_url, _ = normalize_image_path_to_url(image_path_str, domain)
                        
                        if not image_url:
                            continue
                        
                        # Create basic image metadata
                        img_info = {
                            'image_path': image_path_str,
                            'image_url': image_url,
                            'image_id': image_file.stem,
                            'page': page_num,
                            'filename': f"{filename_base}.pdf",
                            'ocr_text': '',  # Will be populated if available
                            'captions': [],
                            'caption_details': [],
                            'figure_number': None,
                            'caption': None,
                            'image_size': [0, 0],
                            'format': 'PNG',
                            'visual_description': self._get_enhanced_image_description({'image_path': image_path_str, 'ocr_text': ''}),
                            'page_source': 'inferred'
                        }
                        
                        # Enrich metadata using page text if available (adds figure numbers/captions)
                        try:
                            if retriever:
                                self._enrich_image_metadata_from_text(
                                    img_info=img_info,
                                    retriever=retriever,
                                    filename_base=filename_base,
                                    page_num=page_num
                                )
                        except Exception as enrich_error:
                            agent_logger.debug(f"   Could not enrich metadata for {image_path_str}: {enrich_error}")
                        
                        if img_info.get('page_source') != 'metadata':
                            img_info['page'] = None
                        
                        # ✅ RAG-BASED: Trust semantic relevance - images from same document are relevant
                        # For file system images, we don't have rich metadata yet, but if from same document, include it
                        # Only require that image has content
                        if image_has_content(img_info):
                            adjacent_images.append(img_info)
                            agent_logger.info(f"   ✅ Found adjacent page image: {image_url} (page {page_num})")
                        else:
                            agent_logger.debug(f"   ⏭️ Skipped adjacent page image: no content (page {page_num})")
                    
                    except (ValueError, IndexError) as e:
                        # Skip files that don't match expected naming pattern
                        agent_logger.debug(f"   Skipped image file (name parsing error): {image_file.name}")
                        continue
            
            # Combine images from vector store (with rich metadata) and images directory (fallback)
            # Vector store images have priority (they have captions/figure numbers)
            all_adjacent_images = adjacent_images_from_store + adjacent_images
            
            # Remove duplicates based on image_path
            seen_paths = set()
            unique_images = []
            for img in all_adjacent_images:
                img_path = img.get('image_path', '').replace('\\', '/')
                if img_path and img_path not in seen_paths:
                    seen_paths.add(img_path)
                    unique_images.append(img)
            
            return unique_images
            
        except Exception as e:
            agent_logger.warning(f"Error finding adjacent page images: {e}")
            return []
    
    def _enrich_image_metadata_from_text(
        self,
        img_info: Dict[str, Any],
        retriever: Any,
        filename_base: str,
        page_num: int
    ) -> None:
        """Populate missing caption/figure metadata using vectorstore text for the page."""
        if not retriever:
            return
        
        try:
            documents_text: List[str] = []
            full_filename = f"{filename_base}.pdf"
            
            if hasattr(retriever, "vectorstore") and retriever.vectorstore:
                try:
                    results = retriever.vectorstore.get(
                        where=build_page_filter(full_filename, page_num),
                        limit=10
                    )
                    documents_text = results.get("documents", []) or []
                except Exception as vector_error:
                    agent_logger.debug(f"   Could not fetch documents for enrichment ({full_filename} page {page_num}): {vector_error}")
            else:
                try:
                    docs = retriever.similarity_search(
                        query=f"Figure captions for page {page_num}",
                        k=6,
                        filter_dict={"file_name": full_filename, "page": page_num},
                        use_cache=False
                    )
                    documents_text = [doc.page_content for doc in docs if hasattr(doc, "page_content")]
                except Exception as similarity_error:
                    agent_logger.debug(f"   Similarity search failed for enrichment ({full_filename} page {page_num}): {similarity_error}")
            
            combined_text = " ".join(documents_text).strip()
            if not combined_text:
                return
            
            # Normalize whitespace to simplify regex matching
            normalized_text = re.sub(r'\s+', ' ', combined_text)
            
            # Extract figure captions: capture the figure number and text until the next figure reference
            figure_pattern = re.compile(
                r'(?:Figure|FIGURE)\s+(\d+(?:\.\d+)?[A-Za-z]?)\s*[:\.\-]?\s*(.+?)(?=(?:Figure|FIGURE)\s+\d|\Z)',
                re.IGNORECASE
            )
            
            matches = list(figure_pattern.finditer(normalized_text))
            if not matches:
                if normalized_text:
                    snippet = normalized_text[:200].strip()
                    if snippet:
                        if not img_info.get("caption"):
                            img_info["caption"] = snippet
                        if not img_info.get("visual_description"):
                            img_info["visual_description"] = snippet
                        if not img_info.get("caption_details"):
                            img_info["caption_details"] = [{
                                "figure_number": None,
                                "caption_text": snippet,
                                "full_caption": snippet
                            }]
                        # Try to infer figure number from snippet
                        figure_match = re.search(r'(?:Figure|FIGURE)\s+(\d+(?:\.\d+)?[A-Za-z]?)', snippet, re.IGNORECASE)
                        if figure_match and not img_info.get("figure_number"):
                            img_info["figure_number"] = figure_match.group(1).strip()
                return
            
            caption_details = []
            for match in matches:
                figure_number = match.group(1).strip()
                caption_text = match.group(2).strip()
                # Clean up repeated "Figure" words in caption text
                caption_text = re.sub(r'^(?:Figure|FIGURE)\s+\d+(?:\.\d+)?[A-Za-z]?\s*[:\.\-]?\s*', '', caption_text, flags=re.IGNORECASE)
                caption_text = caption_text.strip()
                caption_details.append({
                    "figure_number": figure_number,
                    "caption_text": caption_text,
                    "full_caption": f"Figure {figure_number} {caption_text}".strip()
                })
            
            if not caption_details:
                # Build per-page sentence pool for fallback captions
                sentences = tracker.get("sentences")
                if not sentences:
                    sentences = []
                    # Split text into sentences and keep meaningful ones
                    raw_segments = re.split(r'(?<=[\.\?\!])\s+', normalized_text)
                    for segment in raw_segments:
                        segment = segment.strip()
                        if len(segment) >= 40:
                            sentences.append(segment)
                    if not sentences and normalized_text:
                        cleaned_snippet = normalized_text[:200].strip()
                        if cleaned_snippet:
                            sentences = [cleaned_snippet]
                    tracker["sentences"] = sentences
                    tracker["sentence_index"] = 0

                if sentences:
                    idx = tracker.get("sentence_index", 0)  # separate sentence pointer
                    sentence = sentences[idx % len(sentences)]
                    tracker["sentence_index"] = idx + 1

                    if not img_info.get("caption"):
                        img_info["caption"] = sentence
                    if not img_info.get("visual_description"):
                        img_info["visual_description"] = sentence
                    if not img_info.get("captions"):
                        img_info["captions"] = [sentence]
                    if not img_info.get("caption_details"):
                        img_info["caption_details"] = [{
                            "figure_number": img_info.get("figure_number"),
                            "caption_text": sentence,
                            "full_caption": sentence
                        }]
                    # Try to infer figure number from the sentence
                    figure_match = re.search(r'(?:Figure|FIGURE)\s+(\d+(?:\.\d+)?[A-Za-z]?)', sentence, re.IGNORECASE)
                    if figure_match and not img_info.get("figure_number"):
                        img_info["figure_number"] = figure_match.group(1).strip()
                return
            
            key = (filename_base.lower(), page_num)
            tracker = self._page_caption_trackers.setdefault(
                key,
                {"details": caption_details, "index": 0, "sentences": [], "sentence_index": 0}
            )
            # If details changed since last time, reset index
            if tracker["details"] != caption_details:
                tracker["details"] = caption_details
                tracker["index"] = 0
                tracker["sentences"] = []
                tracker["sentence_index"] = 0
            
            # Try to match figure number to assign the correct caption
            figure_number = (img_info.get("figure_number") or "").strip()
            selected_detail = None
            if figure_number:
                for detail in caption_details:
                    if detail.get("figure_number"):
                        if detail["figure_number"].strip().lower() == figure_number.lower():
                            selected_detail = detail
                            break
                # If no exact match, try to match figure keywords to captions
                if not selected_detail:
                    fig_lower = figure_number.lower()
                    for detail in caption_details:
                        text = (detail.get("caption_text") or "").lower()
                        if text and fig_lower in text:
                            selected_detail = detail
                            break
                # Final fallback: use next sequential caption
                if not selected_detail and caption_details:
                    idx = tracker["index"]
                    if idx < len(tracker["details"]):
                        selected_detail = tracker["details"][idx]
                        tracker["index"] += 1
                    else:
                        selected_detail = caption_details[tracker["index"] - 1] if tracker["index"] > 0 else caption_details[0]
            else:
                # No figure number - assign next available caption for this page
                idx = tracker["index"]
                if idx < len(tracker["details"]):
                    selected_detail = tracker["details"][idx]
                    tracker["index"] += 1
                else:
                    selected_detail = caption_details[-1]
                    tracker["index"] = len(caption_details)
                if selected_detail.get("figure_number"):
                    img_info["figure_number"] = selected_detail["figure_number"]
            
            # Populate caption fields using selected detail
            if selected_detail and selected_detail.get("caption_text"):
                caption_text = selected_detail["caption_text"]
                if not img_info.get("caption"):
                    img_info["caption"] = caption_text
                # Include only caption texts (avoid figure numbers) if we already have them
                if not img_info.get("captions"):
                    img_info["captions"] = [
                        detail["caption_text"]
                        for detail in caption_details
                        if detail.get("caption_text")
                    ]
                if not img_info.get("caption_details"):
                    img_info["caption_details"] = caption_details
                current_desc = img_info.get("visual_description")
                if not current_desc or current_desc.strip().lower() in {"", "wide diagram or chart"}:
                    img_info["visual_description"] = caption_text
            else:
                # Fallback to sentence pool if selected detail lacks caption text
                sentences = tracker.get("sentences")
                if not sentences:
                    sentences = []
                    raw_segments = re.split(r'(?<=[\.\?\!])\s+', normalized_text)
                    figure_segments: List[str] = []
                    other_segments: List[str] = []
                    for segment in raw_segments:
                        segment = segment.strip()
                        if len(segment) < 20:
                            continue
                        if 'figure' in segment.lower():
                            figure_segments.append(segment)
                        else:
                            other_segments.append(segment)
                    if figure_segments:
                        sentences.extend(figure_segments)
                    if other_segments:
                        sentences.extend(other_segments)
                    if not sentences and normalized_text:
                        cleaned_snippet = normalized_text.strip()
                        # take first sentence up to period
                        period_idx = cleaned_snippet.find('.')
                        if period_idx != -1:
                            cleaned_snippet = cleaned_snippet[:period_idx + 1]
                        sentences = [cleaned_snippet[:200]]
                    tracker["sentences"] = sentences
                    tracker["sentence_index"] = tracker.get("sentence_index", 0)

                if sentences:
                    idx = tracker.get("sentence_index", 0)
                    sentence = sentences[idx % len(sentences)]
                    tracker["sentence_index"] = idx + 1

                    if not img_info.get("caption"):
                        img_info["caption"] = sentence
                    current_desc = img_info.get("visual_description")
                    if not current_desc or current_desc.strip().lower() in {"", "wide diagram or chart"}:
                        img_info["visual_description"] = sentence
                    if not img_info.get("captions"):
                        img_info["captions"] = [sentence]
                    if not img_info.get("caption_details"):
                        img_info["caption_details"] = [{
                            "figure_number": img_info.get("figure_number"),
                            "caption_text": sentence,
                            "full_caption": sentence
                        }]

            # Ensure visual_description aligns with final caption when available
            final_caption = img_info.get("caption")
            if final_caption:
                current_desc = img_info.get("visual_description")
                if not current_desc or current_desc.strip().lower() in {"", "wide diagram or chart"} or current_desc == final_caption:
                    img_info["visual_description"] = final_caption
        
        except Exception as e:
            agent_logger.debug(f"   Failed to enrich image metadata from text for {img_info.get('image_path')}: {e}")
            return

    def _get_enhanced_image_description(self, img: Dict[str, Any]) -> str:
        """Get enhanced image description using semantic analysis of captions and OCR.
        
        Uses intelligent text analysis instead of hard-coded keyword matching.
        Prioritizes caption text which usually contains the most accurate description.
        """
        try:
            ocr_text = img.get('ocr_text', '')
            caption = img.get('caption', '')
            captions = img.get('captions', [])
            figure_number = img.get('figure_number', '')
            
            # ✅ SMART: Use caption text directly (most accurate source)
            # Captions are written by authors and contain precise descriptions
            best_description = None
            
            if caption and caption.strip():
                best_description = caption.strip()
            elif captions and len(captions) > 0:
                # Use first caption (usually most complete)
                first_caption = captions[0] if isinstance(captions, list) else str(captions)
                if first_caption and first_caption.strip():
                    best_description = first_caption.strip()
            
            # ✅ SMART: If we have caption, use it (it's already well-written)
            if best_description:
                # Add figure number if available and not already in caption
                if figure_number and f"figure {figure_number.lower()}" not in best_description.lower():
                    enhanced = f"Figure {figure_number}: {best_description}"
                else:
                    enhanced = best_description
                
                # Only enhance if description is truly generic (single word)
                generic_words = ['diagram', 'illustration', 'image', 'figure', 'picture']
                if enhanced.lower().strip() in generic_words:
                    # Try to extract more context from OCR
                    if ocr_text and len(ocr_text.strip()) > 20:
                        enhanced = f"{enhanced}: {ocr_text[:150]}"
                else:
                    # Caption is good enough - return as-is
                    return enhanced
            
            # ✅ SMART: If no caption, use OCR with intelligent truncation
            if ocr_text and len(ocr_text.strip()) > 10:
                # Extract meaningful portion (first sentence or first 200 chars)
                # Look for sentence boundaries
                sentences = ocr_text.split('.')
                if len(sentences) > 0 and len(sentences[0].strip()) > 20:
                    enhanced = sentences[0].strip() + '.'
                else:
                    enhanced = ocr_text[:200].strip()
                    if len(ocr_text) > 200:
                        enhanced += '...'
                
                if figure_number:
                    enhanced = f"Figure {figure_number}: {enhanced}"
                
                return enhanced
            
            # ✅ SMART: Fallback - use image descriptor if available
            image_path = img.get('image_path', '')
            if image_path and Path(image_path).exists():
                try:
                    enhanced = self.image_descriptor.enhance_ocr_with_description(image_path, ocr_text)
                    if enhanced and enhanced.lower() not in ['diagram', 'illustration', 'image']:
                        return enhanced
                except:
                    pass
            
            # Final fallback
            return img.get('caption', '') or img.get('ocr_text', '') or "Image"
                
        except Exception as e:
            agent_logger.warning(f"Failed to enhance image description: {e}")
            # Fallback to caption or OCR
            return img.get('caption', '') or img.get('ocr_text', '') or "Image"
    
    def clear_memory(self, session_id: Optional[str] = None) -> None:
        """Clear the conversation memory for a session or all sessions."""
        if session_id:
            if session_id in self.conversation_memory:
                self.conversation_memory[session_id].clear()
                agent_logger.info(f"Cleared conversation memory for session: {session_id}")
        else:
            # Clear all sessions
            self.conversation_memory.clear()
            agent_logger.info("Cleared all conversation memory")


def create_meetara_agent(llm=None) -> MeetaraAgent:
    """Factory function to create a MeetaraAgent instance."""
    return MeetaraAgent()


# Global agent instance (will be initialized when LLM is available)
_meetara_agent: Optional[MeetaraAgent] = None


def get_meetara_agent() -> Optional[MeetaraAgent]:
    """Get the global MeetaraAgent instance."""
    if _meetara_agent is None:
        # Initialize without LLM for offline operation
        initialize_meetara_agent(None)
    
    return _meetara_agent


def initialize_meetara_agent(llm) -> MeetaraAgent:
    """Initialize the global MeetaraAgent instance."""
    global _meetara_agent
    _meetara_agent = create_meetara_agent(llm)
    agent_logger.info("Global MeetaraAgent initialized")
    return _meetara_agent 