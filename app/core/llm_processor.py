"""
Smart LLM Processor for Meetara Core
Integrates with local HuggingFace models for true intelligence
"""

import os
import torch
import signal
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from app.core.logger import agent_logger
from app.core.config import Settings
from app.core.gguf_llm_processor import get_meetara_gguf_processor
import time

class SmartLLMProcessor:
    """Smart LLM processor using local HuggingFace models for true intelligence."""
    
    def __init__(self):
        self.config = Settings()
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        self._model_cache = {}  # Cache for model instances
        self.gguf_processor = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize GGUF models only - no HuggingFace fallback needed."""
        # Initialize GGUF processor (our primary and only LLM)
        if self.config.use_meetara_models:
            try:
                self.gguf_processor = get_meetara_gguf_processor()
                agent_logger.info("✅ Meetara GGUF models initialized (Qwen3-4B Instruct + Thinking)")
            except Exception as e:
                agent_logger.error(f"Failed to initialize GGUF models: {e}")
                raise Exception("GGUF models are required - no fallback available")
        else:
            agent_logger.warning("⚠️ Meetara GGUF models disabled in config - system will not function properly")
        
        agent_logger.info("🚀 LLM initialization complete - using GGUF models only")
    
    def _load_model(self, task: str, model_name: str, model_path: Path, pipeline_task: str):
        """Load a specific model for a task."""
        try:
            if pipeline_task == "text-generation":
                # For Microsoft DialoGPT models
                agent_logger.info(f"Loading tokenizer for {model_name}")
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    use_fast=True  # Use fast tokenizer for Microsoft models
                )
                
                # Add padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                agent_logger.info(f"Loading model for {model_name}")
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    low_cpu_mem_usage=True
                )
                
                # Create generation pipeline
                agent_logger.info(f"Creating pipeline for {model_name}")
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=self.config.local_llm_max_length,
                    do_sample=self.config.local_llm_do_sample,
                    top_p=self.config.local_llm_top_p,
                    top_k=self.config.local_llm_top_k,
                    temperature=self.config.llm_temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                self.models[task] = model
                self.tokenizers[task] = tokenizer
                self.pipelines[task] = generator
                self._model_cache[task] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "pipeline": generator
                }
                agent_logger.info(f"Successfully loaded {model_name} for {task}")
                
        except Exception as e:
            agent_logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def generate_intelligent_response(
        self, 
        query: str, 
        context_docs: List[str], 
        domain: str = "general", 
        emotion: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        associated_images: Optional[List[Dict[str, Any]]] = None  # ✅ Images from documents
    ) -> str:
        """Generate intelligent response using Meetara GGUF models ONLY."""
        try:
            agent_logger.info(f"Generating response for query: {query[:50]}...")
            
            # Use GGUF models (our only LLM)
            if not self.gguf_processor:
                raise Exception("GGUF processor not initialized - cannot generate response")
            
            context = self._prepare_context(context_docs)
            result = self.gguf_processor.generate_response(
                query=query,
                context=context,
                domain=domain,
                emotion=emotion,
                use_thinking=False,  # Always use instruct model (faster, good enough for all domains)
                conversation_history=conversation_history,  # ✅ Pass conversation history
                document_metadata=document_metadata,  # ✅ Pass metadata for source citations
                associated_images=associated_images  # ✅ Pass images so LLM knows about them
            )
            
            if result["success"]:
                response_text = result["response"]
                agent_logger.info(f"✅ Response generated using Meetara GGUF models ({len(response_text)} chars)")
                
                # ✅ Use response formatter to structure the response
                from app.core.response_formatter import get_response_formatter
                formatter = get_response_formatter()
                formatted = formatter.format_response_with_tables(response_text)
                
                # Check if response is structured (has title, sections, etc.)
                has_structure = "**" in response_text and response_text.count("**") >= 2
                if not has_structure:
                    agent_logger.warning(f"⚠️ Response may not be structured. Preview: {response_text[:200]}...")
                
                return formatted.get("formatted_text", response_text)
            else:
                error_msg = result.get("error", "Unknown error")
                agent_logger.error(f"❌ GGUF models failed: {error_msg}")
                agent_logger.warning(f"⚠️ Falling back to content extraction for domain: {domain}")
                return self._extract_intelligent_content(query, context_docs, domain)
                
        except Exception as e:
            agent_logger.error(f"Error in generate_intelligent_response: {e}")
            return self._extract_intelligent_content(query, context_docs, domain)
    
    def _cross_validate_and_refine(self, query: str, response: str, context_docs: List[str]) -> str:
        """Cross-validate and refine the LLM response for accuracy and relevance."""
        try:
            # 1. Check if response actually answers the query
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Extract key terms from query
            query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 3]
            
            # Check if response contains query terms or related concepts
            relevance_score = 0
            for term in query_terms:
                if term in response_lower:
                    relevance_score += 1
            
            # 2. Check if response is too generic
            generic_phrases = [
                "i don't know", "i'm not sure", "i can't answer", "no information",
                "based on the available information", "general in nature"
            ]
            is_generic = any(phrase in response_lower for phrase in generic_phrases)
            
            # 3. Check if response length is appropriate
            is_too_short = len(response.strip()) < 50
            
            # 4. Validate against context documents
            context_relevance = self._validate_against_context(query, response, context_docs)
            
            # Decision logic
            if relevance_score < 2 or is_generic or is_too_short or not context_relevance:
                agent_logger.warning(f"Response failed validation - relevance: {relevance_score}, generic: {is_generic}, short: {is_too_short}, context_relevant: {context_relevance}")
                # Use intelligent extraction instead
                return self._extract_intelligent_content(query, context_docs)
            
            # 5. Refine response if needed
            refined_response = self._refine_response(query, response, context_docs)
            
            agent_logger.info(f"Response validated and refined successfully")
            return refined_response
            
        except Exception as e:
            agent_logger.error(f"Error in cross-validation: {e}")
            return self._extract_intelligent_content(query, context_docs)
    
    def _validate_against_context(self, query: str, response: str, context_docs: List[str]) -> bool:
        """Validate if response is consistent with retrieved context documents."""
        try:
            query_lower = query.lower()
            response_lower = response.lower()
            
            # Check if response mentions concepts that appear in context
            context_text = " ".join([doc.lower() for doc in context_docs])
            
            # Extract key concepts from query
            query_concepts = set(query_lower.split())
            
            # Check if response contains concepts from context
            context_concepts = set(context_text.split())
            response_concepts = set(response_lower.split())
            
            # Find overlap between response and context
            response_context_overlap = response_concepts.intersection(context_concepts)
            
            # If response has significant overlap with context, it's likely relevant
            if len(response_context_overlap) >= 3:
                return True
            
            # Check if response addresses the specific query topic
            if any(concept in response_lower for concept in query_concepts):
                return True
            
            return False
            
        except Exception as e:
            agent_logger.error(f"Error in context validation: {e}")
            return False
    
    def _refine_response(self, query: str, response: str, context_docs: List[str]) -> str:
        """Refine the response to be more accurate and complete."""
        try:
            # If response is good, return as is
            if len(response.strip()) > 100 and "based on the available information" not in response.lower():
                return response
            
            # Otherwise, enhance with intelligent extraction
            enhanced_content = self._extract_intelligent_content(query, context_docs)
            
            # Combine LLM response with extracted content
            if response and len(response.strip()) > 20:
                combined_response = f"{response}\n\n{enhanced_content}"
                return combined_response
            else:
                return enhanced_content
                
        except Exception as e:
            agent_logger.error(f"Error in response refinement: {e}")
            return response
    
    def _extract_intelligent_content(self, query: str, context_docs: List[str], domain: str = "general") -> str:
        """Extract and format intelligent content from documents with enhanced relevance.
        
        ⚠️ WARNING: This is a FALLBACK method when LLM fails. It should NOT be used for normal responses.
        All responses should go through the LLM for proper structure and domain-appropriate formatting.
        """
        if not context_docs:
            return "I understand your question. While I don't have specific information about this topic in my current knowledge base, I'd be happy to help you find relevant resources."
        
        agent_logger.warning(f"⚠️ Using fallback content extraction (LLM failed) for domain: {domain}")
        
        # Extract key information based on query with enhanced relevance
        query_lower = query.lower()
        query_terms = [term.strip() for term in query_lower.split() if len(term.strip()) > 2]
        relevant_content = []
        
        for doc in context_docs[:3]:  # Use top 3 documents
            # Extract sentences that might be relevant
            sentences = doc.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30:  # Only meaningful sentences
                    sentence_lower = sentence.lower()
                    
                    # Calculate relevance score for this sentence based on contextual matching
                    relevance_score = 0
                    
                    # 1. Exact term matches (highest weight)
                    for term in query_terms:
                        if term in sentence_lower:
                            relevance_score += 5
                    
                    # 2. Phrase matches (medium weight)
                    query_phrases = query_lower.split()
                    if len(query_phrases) > 1:
                        for i in range(len(query_phrases) - 1):
                            phrase = f"{query_phrases[i]} {query_phrases[i+1]}"
                            if phrase in sentence_lower:
                                relevance_score += 4
                    
                    # 3. Semantic similarity (lower weight)
                    if any(word in sentence_lower for word in query_terms):
                        relevance_score += 2
                    
                    # 4. Contextual relevance bonus
                    query_concepts = set(query_lower.split())
                    sentence_concepts = set(sentence_lower.split())
                    common_concepts = query_concepts.intersection(sentence_concepts)
                    if len(common_concepts) >= 2:  # At least 2 common words
                        relevance_score += 3
                    
                    # Only include sentences with meaningful relevance
                    if relevance_score >= 3:
                        relevant_content.append((sentence, relevance_score))
        
        if relevant_content:
            # Sort by relevance and take top 5 (more content for fallback)
            relevant_content.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [sentence for sentence, score in relevant_content[:5]]
            
            # ✅ DOMAIN-AWARE FORMATTING: Only add medical disclaimer for medical domains
            # Format the response with better structure
            response = f"Based on the available information from the knowledge base:\n\n"
            response += "\n\n".join([f"• {sentence}" for sentence in top_sentences])
            
            # ✅ Only add medical disclaimer for actual medical/health domains
            domain_lower = domain.lower()
            is_medical_domain = any(d in domain_lower for d in ["health", "medical", "nutrition", "mental", "medication", "chronic", "preventive", "women_health", "senior_health"])
            
            if is_medical_domain:
                response += "\n\n⚠️ **Important Medical Disclaimer:**\n• This information is general in nature\n• Always consult qualified healthcare professionals\n• Medical decisions require professional evaluation"
            else:
                # For non-medical domains, add appropriate disclaimer or skip
                if domain_lower in ["legal", "legal_assistance"]:
                    response += "\n\n⚠️ **Important:** This information is general in nature. For specific legal advice, consult with a qualified attorney."
                elif domain_lower in ["financial", "financial_planning"]:
                    response += "\n\n⚠️ **Important:** This information is general in nature. For personalized financial advice, consult with a certified financial planner."
            
            return response
        else:
            return "I understand your question. While I don't have specific information about this topic in my current knowledge base, I'd be happy to help you find relevant resources or answer other questions you might have."
    
    def _prepare_context(self, context_docs: List[str]) -> str:
        """Prepare context from documents."""
        if not context_docs:
            return "No specific context available."
        
        # Combine all context documents
        context_text = "\n\n".join([doc.strip() for doc in context_docs if doc.strip()])
        return context_text[:2000]  # Limit context length
    
    def _create_intelligent_prompt(self, query: str, context: str) -> str:
        """Create a prompt for Microsoft DialoGPT model."""
        # Microsoft DialoGPT uses a simple conversation format
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        return prompt
    
    def _generate_with_llm(self, prompt: str, query: str, context_docs: List[str]) -> str:
        """Generate response using local LLM with timeout."""
        try:
            generator = self.pipelines["conversation"]
            
            agent_logger.info("Starting LLM generation...")
            
            def generate_response():
                # Generate response with optimized parameters for speed
                outputs = generator(
                    prompt,
                    max_new_tokens=50,  # Shorter for faster responses
                    do_sample=True,
                    temperature=0.5,  # Lower for more focused responses
                    top_p=0.8,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizers["conversation"].eos_token_id,
                    eos_token_id=self.tokenizers["conversation"].eos_token_id,
                    num_beams=1,  # Greedy decoding for speed
                    use_cache=True
                )
                
                # Extract generated text
                generated_text = outputs[0]["generated_text"]
                
                # Extract only the new generated part (after the prompt)
                if len(generated_text) > len(prompt):
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text.strip()
                
                # Clean up the response
                response = self._post_process_response(response)
                    
                return response
            
            # Run with 30-second timeout for complex queries (increased from 10s)
            response = run_with_timeout(generate_response, timeout_seconds=30)
            
            if response is None:
                agent_logger.warning("LLM generation timed out, using intelligent extraction")
                return self._extract_intelligent_content(query, context_docs)
            
            agent_logger.info("LLM generation completed successfully")
            
            # Check if response is valid
            if response and len(response.strip()) > 10:
                return response
            else:
                agent_logger.warning("LLM generated empty or invalid response, using intelligent extraction")
                return self._extract_intelligent_content(query, context_docs)
            
        except Exception as e:
            agent_logger.error(f"LLM generation error: {e}")
            return self._extract_intelligent_content(query, context_docs)
    
    def _post_process_response(self, response: str) -> str:
        """Post-process LLM response with validation."""
        # Clean up the response
        response = response.strip()
        
        # Remove any remaining prompt artifacts
        if response.startswith("assistant"):
            response = response[9:].strip()
        
        # Check if response is empty or just contains the prompt
        if not response or len(response) < 10:
            return None  # Signal to use fallback
        
        # Check if response is too generic
        generic_phrases = ["i don't know", "i'm not sure", "i can't answer", "no information"]
        if any(phrase in response.lower() for phrase in generic_phrases):
            return None  # Signal to use fallback
        
        return response
        
        # Ensure response is not empty
        if not response:
            return "I understand your question. Let me provide you with some helpful information based on the available context."
        
        return response
    
    def _fallback_response(self, query: str, context_docs: List[str]) -> str:
        """Provide fallback response when LLM fails."""
        if context_docs:
            # Extract key information from context
            context_summary = " ".join([doc[:200] for doc in context_docs[:2]])
            return f"Based on the available information: {context_summary[:300]}... Please consult with a healthcare professional for personalized advice."
        else:
            return "I understand your question. While I don't have specific information about this topic in my current knowledge base, I'd be happy to help you find relevant resources or answer other questions you might have."
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotion in text using local emotion models."""
        if "emotion" not in self.pipelines:
            return {"emotion": "neutral", "confidence": 0.5}
        
        try:
            classifier = self.pipelines["emotion"]
            result = classifier(text)
            
            return {
                "emotion": result[0]["label"],
                "confidence": result[0]["score"],
                "all_emotions": result
            }
        except Exception as e:
            agent_logger.error(f"Emotion analysis error: {e}")
            return {"emotion": "neutral", "confidence": 0.5}
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for texts using local embedding model."""
        if "embedding" not in self.models:
            agent_logger.warning("No embedding model available")
            return [[0.0] * 384] * len(texts)  # Default embedding size
        
        try:
            embedder = self.models["embedding"]
            embeddings = embedder.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            agent_logger.error(f"Embedding error: {e}")
            return [[0.0] * 384] * len(texts)

# Global instance
_llm_processor = None

def timeout_handler(signum, frame):
    """Timeout handler for LLM generation."""
    raise TimeoutError("LLM generation timed out")

def run_with_timeout(func, timeout_seconds=10):
    """Run a function with timeout."""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        agent_logger.warning(f"LLM generation timed out after {timeout_seconds} seconds")
        return None
    
    if exception[0]:
        raise exception[0]
    
    return result[0]

def get_llm_processor() -> SmartLLMProcessor:
    """Get global LLM processor instance."""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = SmartLLMProcessor()
    return _llm_processor 