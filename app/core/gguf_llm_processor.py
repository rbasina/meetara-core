"""
GGUF LLM Processor for Meetara Core
Uses custom fine-tuned GGUF models for optimal performance
Supports automatic download from Hugging Face with caching
"""

import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from pathlib import Path
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, snapshot_download
from app.core.logger import agent_logger
from app.core.config import Settings
from app.core.domain_categorizer import get_domain_categorizer  # ✅ Optimized domain categorization
from app.core.config_loader import config_loader  # ✅ Use config for fallback messages


class MeetaraGGUFProcessor:
    """GGUF LLM processor using custom fine-tuned Meetara models."""
    
    def __init__(self):
        self.config = Settings()
        self.models = {}  # Will store loaded models
        self.model_paths = {}  # Will store paths for lazy loading
        self._model_cache = {}
        self.domain_categorizer = get_domain_categorizer()  # ✅ Use optimized categorizer
        self._initialize_models()
        
    def _download_model_from_hf(self, model_id: str, filename: str) -> Optional[Path]:
        """Download model from Hugging Face and cache it locally.
        
        Uses standard Hugging Face cache location: ~/.cache/huggingface/hub
        On Windows: C:/Users/<username>/.cache/huggingface/hub
        
        Returns:
            Path to downloaded model file, or None if download failed
        """
        try:
            # Determine cache location
            if self.config.meetara_model_cache_dir:
                # Custom cache directory
                cache_dir = self.config.meetara_model_cache_dir
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_location = str(cache_dir)
                
                # Check custom cache location first
                cached_path = cache_dir / filename
                if cached_path.exists():
                    file_size_gb = cached_path.stat().st_size / (1024**3)
                    agent_logger.info(f"✅ Model found in custom cache: {cached_path.name} ({file_size_gb:.2f} GB)")
                    agent_logger.info(f"   Cache location: {cache_location}")
                    return cached_path
            else:
                # Use default HF cache location
                cache_location = str(Path.home() / ".cache" / "huggingface" / "hub")
                agent_logger.info(f"   Using default HF cache: {cache_location}")
            
            # Download from HF (hf_hub_download automatically checks cache first)
            agent_logger.info(f"📥 Downloading model from Hugging Face: {model_id}/{filename}")
            agent_logger.info(f"   Cache location: {cache_location}")
            agent_logger.info(f"   This may take a few minutes (model size: ~1.2 GB)...")
            
            start_time = time.time()
            
            # Build download parameters
            download_kwargs = {
                "repo_id": model_id,
                "filename": filename
            }
            
            if self.config.meetara_model_cache_dir:
                # Custom cache directory
                download_kwargs["cache_dir"] = cache_location
                download_kwargs["local_dir"] = cache_location
            # else: use default HF cache (don't specify cache_dir/local_dir)
            
            downloaded_path = hf_hub_download(**download_kwargs)
            
            download_time = time.time() - start_time
            downloaded_file = Path(downloaded_path)
            
            if downloaded_file.exists():
                file_size_gb = downloaded_file.stat().st_size / (1024**3)
                agent_logger.info(f"✅ Model downloaded successfully in {download_time:.1f}s ({file_size_gb:.2f} GB)")
                agent_logger.info(f"   Cached at: {downloaded_path}")
                return downloaded_file
            else:
                agent_logger.warning(f"⚠️ Downloaded file not found at expected path: {downloaded_path}")
                return None
            
        except Exception as e:
            agent_logger.error(f"❌ Failed to download model from Hugging Face: {e}")
            agent_logger.warning(f"   Falling back to local model path if available")
            return None
    
    def _initialize_models(self):
        """Initialize custom Meetara GGUF models with LAZY LOADING.
        
        Supports both Hugging Face downloads and local model paths.
        """
        if not self.config.use_meetara_models:
            agent_logger.info("Meetara models disabled, using fallback")
            return
        
        # Priority 1: Try Hugging Face model if configured
        if self.config.meetara_hf_model_id:
            agent_logger.info(f"🔍 Checking Hugging Face model: {self.config.meetara_hf_model_id}")
            
            downloaded_path = self._download_model_from_hf(
                self.config.meetara_hf_model_id,
                self.config.meetara_hf_model_file
            )
            
            if downloaded_path and downloaded_path.exists():
                self.model_paths["instruct"] = downloaded_path
                file_size_gb = downloaded_path.stat().st_size / (1024**3)
                agent_logger.info(f"✅ Instruct model ready for lazy loading: {downloaded_path.name} ({file_size_gb:.2f} GB)")
                agent_logger.info("🚀 Lazy loading enabled - model will load only when first needed")
                return
            else:
                agent_logger.warning(f"⚠️ HF model download failed, trying local paths...")
        
        # Priority 2: Fallback to local model paths
        models_path = self.config.meetara_models_path
        
        if not models_path.exists():
            agent_logger.error(f"Meetara models path not found: {models_path}")
            agent_logger.error(f"   Please set MEETARA_HF_MODEL_ID or ensure local models exist")
            return
        
        # Store model paths for lazy loading (don't load yet!)
        instruct_model_path = models_path / self.config.meetara_instruct_model
        if instruct_model_path.exists():
            self.model_paths["instruct"] = instruct_model_path
            file_size_gb = instruct_model_path.stat().st_size / (1024**3)
            agent_logger.info(f"✅ Instruct model ready for lazy loading: {instruct_model_path.name} ({file_size_gb:.2f} GB)")
        else:
            agent_logger.warning(f"Instruct model not found: {instruct_model_path}")
            
        thinking_model_path = models_path / self.config.meetara_thinking_model
        if thinking_model_path.exists():
            self.model_paths["thinking"] = thinking_model_path
            file_size_gb = thinking_model_path.stat().st_size / (1024**3)
            agent_logger.info(f"✅ Thinking model ready for lazy loading: {thinking_model_path.name} ({file_size_gb:.2f} GB)")
        else:
            agent_logger.warning(f"Thinking model not found: {thinking_model_path}")
        
        agent_logger.info("🚀 Lazy loading enabled - models will load only when first needed")
    
    def _load_model_lazy(self, model_type: str) -> bool:
        """Lazy load a model only when needed - saves memory!"""
        if model_type in self.models:
            return True  # Already loaded
        
        if model_type not in self.model_paths:
            agent_logger.error(f"Model path not found for: {model_type}")
            return False
        
        try:
            agent_logger.info(f"🔄 Loading {model_type} model on first use: {self.model_paths[model_type]}")
            start_time = time.time()
            self.models[model_type] = Llama(
                model_path=str(self.model_paths[model_type]),
                n_ctx=self.config.llm_context_length,
                n_threads=os.cpu_count(),
                verbose=False
            )
            load_time = time.time() - start_time
            agent_logger.info(f"✅ {model_type.capitalize()} model loaded in {load_time:.1f}s")
            return True
        except Exception as e:
            agent_logger.error(f"Failed to load {model_type} model: {e}")
            return False
    
    def generate_response(
        self, 
        query: str, 
        context: str = "", 
        domain: str = "general",
        emotion: Optional[str] = None,
        use_thinking: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        associated_images: Optional[List[Dict[str, Any]]] = None  # ✅ Images from documents
    ) -> Dict[str, Any]:
        """Generate response using custom Meetara models with lazy loading.
        
        IMPORTANT: This method ALWAYS uses the LLM model, even when context is empty.
        - Empty context = LLM uses its general knowledge (NO fallback message!)
        - Fallback responses are ONLY used if the model fails to load or throws an error
        - This ensures users always get intelligent responses, not generic fallback messages
        """
        
        # Choose model based on domain and requirements
        model_type = "thinking" if use_thinking else "instruct"
        
        # Lazy load model if not already loaded
        if model_type not in self.models:
            if not self._load_model_lazy(model_type):
                return self._fallback_response(query, context, domain, emotion)
            
        try:
            # Build prompt based on domain and context
            # Note: Empty context is OK - LLM will use general knowledge
            prompt = self._build_prompt(query, context, domain, emotion, model_type, conversation_history, document_metadata, associated_images)
            
            # ✅ DEBUG: Log prompt structure information for troubleshooting
            prompt_length = len(prompt)
            has_structure_instructions = "RESPONSE STRUCTURE" in prompt or "FORMATTING RULES" in prompt
            has_context = context and context != "No specific context available."
            agent_logger.info(f"📝 Prompt built: {prompt_length} chars, Structure instructions: {has_structure_instructions}, Context: {has_context}")
            agent_logger.debug(f"📝 Prompt preview (first 500 chars): {prompt[:500]}...")
            
            # Log context status for debugging
            if not context or context == "No specific context available.":
                agent_logger.info(f"Using LLM with general knowledge (no RAG context) for domain: {domain}")
            else:
                agent_logger.info(f"Using LLM with RAG context ({len(context)} chars) for domain: {domain}")
            
            # Generate response using LLM (always, even if context is empty)
            start_time = time.time()
            max_tokens = self.config.local_llm_max_length
            if domain == "general_knowledge":
                max_tokens = min(max_tokens, 512)
            response = self.models[model_type](
                prompt,
                max_tokens=max_tokens,
                temperature=self.config.llm_temperature,
                top_p=self.config.local_llm_top_p,
                top_k=self.config.local_llm_top_k,
                stop=["</s>", "<|im_end|>", "\n\nHuman:", "\n\nAssistant:", "(End of response)", "Final output:", "🛑", "STOP"],
                echo=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract response text
            response_text = response["choices"][0]["text"].strip()
            
            # ✅ Minimal safety net: Only remove obvious duplicates (prevention should handle most cases)
            # Find first Sources section - response should end there
            first_sources_idx = response_text.find("**Sources**")
            
            if first_sources_idx > 0:
                # Find end of Sources section (after bullet points)
                sources_end = first_sources_idx + len("**Sources**")
                remaining = response_text[sources_end:]
                lines = remaining.split('\n')
                
                # Count Sources bullet points (typically 3-5)
                bullet_count = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('-'):
                        bullet_count += 1
                    elif line.strip() and bullet_count >= 3:
                        # Found end of Sources section
                        sources_section_end = sources_end + sum(len(l) + 1 for l in lines[:i])
                        break
                else:
                    # Sources section continues to end of text
                    sources_section_end = len(response_text)
                
                # If there's content after Sources, it's repetition - truncate it
                if sources_section_end < len(response_text):
                    text_after = response_text[sources_section_end:].strip()
                    if text_after:
                        response_text = response_text[:sources_section_end].strip()
                        agent_logger.warning(f"⚠️ Removed {len(text_after)} chars after Sources (should have been prevented by prompt)")
            
            # Check for duplicate Quick Answer at start (shouldn't happen with proper prompt)
            if response_text.count("**Quick Answer:**") > 1 or response_text.count("Quick Answer:") > 1:
                first_qa = response_text.find("**Quick Answer:**")
                if first_qa < 0:
                    first_qa = response_text.find("Quick Answer:")
                second_qa = response_text.find("**Quick Answer:**", first_qa + 1)
                if second_qa < 0:
                    second_qa = response_text.find("Quick Answer:", first_qa + 10)
                if second_qa > 0:
                    response_text = response_text[:second_qa].strip()
                    agent_logger.warning(f"⚠️ Removed duplicate Quick Answer (should have been prevented by prompt)")
            
            # ✅ DEBUG: Check if response follows structure
            has_title = "**" in response_text and response_text.count("**") >= 2
            has_intro = len(response_text.split('\n')) > 3  # Has multiple lines (likely intro + body)
            has_sources = "**Sources**" in response_text or "Sources:" in response_text or "**Source**" in response_text
            
            # ✅ Check if response uses domain-specific section headers
            section_headers = self._get_domain_category_sections(domain)
            used_sections = []
            for section in section_headers:
                # Check if section header (without **) appears in response
                # Look for both with and without bold markers
                section_text = section.replace("**", "").strip()
                section_with_bold = section.strip()
                
                # Check multiple patterns: exact match, with bold, partial match (key words)
                if (section_text.lower() in response_text.lower() or 
                    section_with_bold.lower() in response_text.lower()):
                    used_sections.append(section_text)
                else:
                    # Try partial match: check if key words from section appear together
                    # e.g., "Core Concepts" from "**Core Concepts & Fundamentals**"
                    key_words = [w for w in section_text.split() if len(w) > 3 and w.lower() not in ['and', 'the', 'for', 'with']]
                    if len(key_words) >= 2:
                        # Check if at least 2 key words appear near each other (within 50 chars)
                        response_lower = response_text.lower()
                        for i, word1 in enumerate(key_words):
                            for word2 in key_words[i+1:]:
                                idx1 = response_lower.find(word1.lower())
                                idx2 = response_lower.find(word2.lower())
                                if idx1 != -1 and idx2 != -1 and abs(idx1 - idx2) < 50:
                                    used_sections.append(section_text)
                                    break
                            if section_text in used_sections:
                                break
            
            structure_score = sum([has_title, has_intro, has_sources])
            sections_used_score = min(len(used_sections), 3)  # Max 3 points for sections
            total_structure_score = structure_score + sections_used_score
            
            agent_logger.info(f"📊 Response structure check: Title={has_title}, Intro={has_intro}, Sources={has_sources}, Sections={len(used_sections)}/{len(section_headers)}, Total score: {total_structure_score}/6")
            agent_logger.info(f"📋 Domain sections used: {used_sections[:3] if used_sections else 'None'} (Expected: {[s.replace('**', '').strip() for s in section_headers[:3]]})")
            
            if total_structure_score < 3 or len(used_sections) == 0:
                agent_logger.warning(f"⚠️ Response may not follow structure! Expected {len(section_headers)} sections, found {len(used_sections)}. Preview: {response_text[:200]}...")
            
            # Log generation performance
            agent_logger.info(f"✅ Generated {len(response_text.split())} tokens in {generation_time:.1f}s ({len(response_text.split())/generation_time:.1f} tokens/sec)")
            
            # Determine context source for logging
            context_source = "RAG documents" if (context and context != "No specific context available.") else "general knowledge"
            
            return {
                "response": response_text,
                "model_used": f"meetara-{model_type}",
                "domain": domain,
                "emotion": emotion,
                "context_used": bool(context and context != "No specific context available."),
                "context_source": context_source,
                "generation_time": round(generation_time, 2),
                "tokens_generated": len(response_text.split()),
                "success": True
            }
            
        except Exception as e:
            # Only use fallback if model generation fails (not for empty context!)
            agent_logger.error(f"Error generating response with {model_type} model: {e}")
            agent_logger.warning("Falling back to domain-specific message due to model error")
            return self._fallback_response(query, context, domain, emotion)
    
    def _get_domain_category_sections(self, domain: str) -> List[str]:
        """Get domain-appropriate section headers based on domain category.
        
        ✅ OPTIMIZED: Uses centralized DomainCategorizer for efficient lookup.
        """
        return self.domain_categorizer.get_domain_sections(domain)
    
    def _get_domain_authorities(self, domain: str) -> str:
        """Get domain-specific authority references.
        
        ✅ OPTIMIZED: Uses centralized DomainCategorizer for efficient lookup.
        """
        return self.domain_categorizer.get_domain_authorities(domain)
    
    def _build_prompt(
        self, 
        query: str, 
        context: str, 
        domain: str, 
        emotion: Optional[str],
        model_type: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        associated_images: Optional[List[Dict[str, Any]]] = None  # ✅ Images from documents
    ) -> str:
        """Build domain-specific prompt for the model with adaptive response structure."""
        
        # Prepare human readable names
        domain_display = domain.replace('_', ' ')
        domain_display_title = domain_display.title()
        
        # Get domain-appropriate sections
        section_headers = self._get_domain_category_sections(domain)
        # Show headers EXACTLY as they should appear (just the headers, no instructions)
        sections_list = "\n".join([f"{section}" for section in section_headers])
        
        # ✅ DEBUG: Log which sections are being used for this domain
        agent_logger.debug(f"📋 Domain '{domain}' sections: {section_headers}")
        
        # Get domain-appropriate authorities
        authorities = self._get_domain_authorities(domain)

        # Pre-compute section text for lightweight reminders
        def build_inline_template(headers: List[str]) -> str:
            lines = ["**Quick Answer:** <direct answer>"]
            for header in headers:
                lines.append(f"{header}")
                lines.append("- key point")
            lines.append("**Conclusion**")
            lines.append("<summary>")
            lines.append("**Sources**")
            lines.append("- Source: <reference>")
            return "\n".join(lines)

        # Determine strictness based on domain safety profile (from meetara_lab_core.py approach)
        strict_structure_domains = {"healthcare", "mental_health", "emergency_care", "chronic_conditions", 
                                   "medication_management", "preventive_care", "women_health", "senior_health",
                                   "general_health", "bipolar_disorder", "stress_management"}
        strict_structure = domain in strict_structure_domains
        
        # Instruction strength depends on domain safety profile (like meetara_lab_core.py)
        if strict_structure:
            format_instruction = "You MUST follow this exact format for ALL responses:"
            structure_note = "For safety-critical domains, follow the structure exactly."
        else:
            format_instruction = "You SHOULD organize your answer using this format:"
            structure_note = "You may adapt section titles if another format better fits the question, but keep the response clear and well-organized."
        
        # Base system prompt with adaptive structure (enhanced from meetara_lab_core.py)
        if domain == "general_knowledge":
            # Show headers EXACTLY as they should appear (just headers, no instructions)
            general_sections = "\n".join([f"{header}" for header in section_headers])
            system_prompt = f"""You are meeTARA, a precise and factual general-knowledge assistant. {format_instruction}

RESPONSE STRUCTURE - Follow this exact format:

**Quick Answer:** [Your direct answer in one sentence with justification]

**[Your Title About the Topic]**
[2-3 sentences introducing the topic and setting context]

AVAILABLE SECTION HEADERS FOR GENERAL KNOWLEDGE (select 2-3 and use them exactly with **bold**):
{general_sections}

EXAMPLE OF COMPLETE RESPONSE FORMAT (using general knowledge domain headers):
**Quick Answer:** The hypotenuse of a right triangle with legs 5 cm and 12 cm is 13 cm, calculated using the Pythagorean theorem.

**Understanding Right Triangles: A Comprehensive Guide**
Right triangles are fundamental geometric shapes with one 90-degree angle. The Pythagorean theorem provides a reliable method to calculate the hypotenuse when the lengths of the two legs are known.

{section_headers[0] if len(section_headers) > 0 else "**Core Concepts & Fundamentals**"}
- The Pythagorean theorem states that in a right triangle, a² + b² = c², where c is the hypotenuse
- A right triangle has one angle measuring exactly 90 degrees
- The hypotenuse is always the longest side, opposite the right angle

{section_headers[1] if len(section_headers) > 1 else "**Practical Application & Practice**"}
- Apply the formula: √(5² + 12²) = √(25 + 144) = √169 = 13 cm
- Verify your answer by checking that 5² + 12² = 13² (25 + 144 = 169)
- Use this method for any right triangle when you know two sides

**Conclusion**
Understanding the Pythagorean theorem enables you to solve right triangle problems efficiently. This fundamental concept is essential for geometry and real-world applications.

**Sources**
- Based on general geometry knowledge and mathematical principles

INSTRUCTIONS:
- Use the structure shown in the example above
- Replace "[Your Title About the Topic]" with an actual title
- Replace "[2-3 sentences...]" with actual introduction text
- Select 2-3 section headers from the available list and use them exactly (with **bold**)
- Write 3-5 bullet points under each section header you choose
- Write a conclusion summarizing key takeaways
- List sources if RAG documents were used, otherwise use "Based on general knowledge and authoritative references"

🛑 STOP IMMEDIATELY AFTER SOURCES:
- After completing the Sources bullet points, STOP - do NOT write anything else
- The response ENDS at Sources - no repetition, no additional sections, no calculations
- Do NOT repeat Quick Answer, Conclusion, or any section after Sources

CRITICAL RULES:
- ❌ ABSOLUTELY NEVER use <think>, <reasoning>, or ANY internal thinking tags
- ❌ ABSOLUTELY NEVER start with meta-commentary like "Okay, I need to..." or "Let me think..."
- ❌ ABSOLUTELY NEVER show your reasoning process or thought patterns
- ❌ ABSOLUTELY NEVER repeat any section after Sources - STOP at Sources
- ❌ Replace ALL placeholders with actual content - do NOT copy text like "[Your Title]" or "[2-3 sentences]"
- ✅ Use the section headers exactly as shown above
- ✅ Do NOT add numbers or labels
- ✅ Generate the response ONCE - do NOT repeat sections
- ✅ STOP IMMEDIATELY after Sources section"""
        else:
            # Use meetara_lab_core.py structure as default, but with domain-specific headers
            # Map domain-specific headers to the meetara_lab_core.py format
            # Use first 3-4 domain-specific headers if available
            selected_headers = section_headers[:min(4, len(section_headers))]
            
            # Build sections following meetara_lab_core.py pattern but with domain headers
            sections_template = ""
            if len(selected_headers) >= 1:
                sections_template += f"{selected_headers[0]}\n"
                sections_template += "[Rich analysis with 2-3 specific data points, research findings, or expert insights. Include numbers/percentages when relevant.]\n\n"
            
            if len(selected_headers) >= 2:
                sections_template += f"{selected_headers[1]}\n"
                sections_template += "1. [First immediate action - specific and achievable]\n"
                sections_template += "2. [Second step - builds on step 1]\n"
                sections_template += "3. [Third step - timeline/follow-up]\n\n"
            
            if len(selected_headers) >= 3:
                sections_template += f"{selected_headers[2]}\n"
                sections_template += "[2-3 practical tips, common pitfalls to avoid, or expert recommendations. Keep it practical and specific.]\n\n"
            
            # Add "Thoughtful Next Question" if we have a 4th header, otherwise use it as optional
            if len(selected_headers) >= 4:
                sections_template += f"{selected_headers[3]}\n"
                sections_template += f"[Ask ONE specific question that offers to provide MORE detailed help specifically within the '{domain_display}' domain. Use natural language patterns like \"Would you like me to create/provide/design a [relevant {domain_display} resource]?\" Make the offer specific to their situation, current progress, and what would be most helpful next in this domain.]\n\n"
            
            system_prompt = f"""You are me²TARA, an advanced AI assistant specialized in {domain_display}. {format_instruction}

**Quick Answer: What This Means for You (Direct Answer)**
[Direct, specific response - 2-4 sentences. Be empathetic and address their specific situation.]

{sections_template}

**Sources**
- [If RAG documents were used: list PDF filenames and page numbers]
- [If no RAG: "Based on general {domain} knowledge"]

CRITICAL GUIDELINES:
- ❌ ABSOLUTELY NEVER use <think>, <reasoning>, or ANY internal thinking tags
- ❌ ABSOLUTELY NEVER start with meta-commentary like "Okay, the user wants..." or "Let me think..." or "The query asks about..."
- ❌ ABSOLUTELY NEVER show your reasoning process or thought patterns
- ❌ ABSOLUTELY NEVER copy bracketed instruction text like "[Direct, specific response...]", "[Rich analysis...]", "[First immediate action...]" - these are INSTRUCTIONS to follow, NOT content to output
- ❌ ABSOLUTELY NEVER copy internal instructions like "[Read their exact question carefully...]" or "[Be emotionally intelligent...]" - these are guidance for you, NOT content for the user
- ❌ Replace ALL bracketed placeholders with actual content - do NOT copy them literally
- ❌ Do NOT add numbers, labels, or meta-commentary
- ✅ ALWAYS read their exact question carefully and understand the HUMAN CONTEXT behind it. Be emotionally intelligent and empathetic.
- ✅ ALWAYS start with a contextual greeting that acknowledges their specific question
- ✅ For safety-critical domains (health, crisis, mental health, legal/financial), follow the structure exactly with domain-specific headers
- ✅ For other domains, you may adapt section titles and structure if another format better fits the question, but keep the response clear and well-organized
- ✅ Use the domain-specific section headers exactly as shown above (copy them exactly with **bold**)
- ✅ Write REAL content under each header - replace all [Write...] instructions with actual text
- ✅ ALWAYS write as if speaking directly to the user about their unique situation
- ✅ Generate the response ONCE - do NOT repeat sections
- ✅ {structure_note}
- ✅ STOP IMMEDIATELY after completing Sources section - the response ENDS at Sources"""
        
        # ✅ Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            history_context = "\n".join([
                f"{'Human' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
                for msg in conversation_history[-6:]  # Last 3 exchanges (6 messages)
            ])
            system_prompt += f"\n\nPrevious conversation:\n{history_context}\n\nPlease continue the conversation naturally, referencing previous context when relevant."
        
        # Add emotion context if available
        if emotion:
            system_prompt += f" The user appears to be feeling {emotion}, so respond with appropriate empathy and support."
        
        # Add context if available
        if context and context != "No specific context available.":
            # Check if context contains tables
            has_tables = "[Table Data]" in context or "[Table:" in context
            
            system_prompt += f"\n\nRelevant context from knowledge base:\n{context}"
            system_prompt += "\n\nImportant: Use the context above if it's relevant to the question. If the context doesn't contain the answer, use your general knowledge to provide a helpful response."
            
            # ✅ CRITICAL: Always preserve exact content from documents (applies to ALL content types)
            # No hardcoding needed - these rules preserve ANY structured content:
            # - Mathematical formulas, equations, theorems (with numbers/labels)
            # - Scientific notation, chemical formulas, technical definitions
            # - Code snippets, algorithms, data structures
            # - Legal citations, medical terminology, academic references
            # - Any numbered/labeled content (sections, figures, tables, etc.)
            system_prompt += "\n\n📐 CRITICAL CONTENT PRESERVATION RULES (Applies to ALL content types):\n"
            system_prompt += "1. If the context contains specific formulas, equations, theorems, definitions, code, or any structured content, you MUST include them EXACTLY as written in the document.\n"
            system_prompt += "2. Preserve ALL reference numbers from the document: theorem numbers (e.g., 'Theorem 4.8'), equation labels (e.g., '(4.29)'), section numbers (e.g., '4.5'), figure numbers, table numbers, etc.\n"
            system_prompt += "3. Use the EXACT notation from the document (mathematical, scientific, technical, legal, medical) - do NOT paraphrase, simplify, or modify formulas/equations/definitions.\n"
            system_prompt += "4. If the document provides specific examples, definitions, or explanations, reference those rather than creating generic alternatives.\n"
            system_prompt += "5. When explaining concepts, quote or closely paraphrase the document's explanations to maintain accuracy and preserve the author's intended meaning.\n"
            system_prompt += "6. DO NOT replace document content with generic examples - use the document's content as the primary and authoritative source."
            
            # Add table formatting instructions if tables are present
            if has_tables:
                system_prompt += "\n\n📊 TABLE FORMATTING INSTRUCTIONS:\n"
                system_prompt += "1. If the context above contains tables (marked with [Table Data] or [Table: table_X_Y]), PRESERVE them in your response as Markdown tables.\n"
                system_prompt += "2. When referencing table data, include the actual Markdown table format in your response so users can see the structured data.\n"
                system_prompt += "3. Example: If context shows '| Region | % |\n|--------|---|\n| Western Europe | 25.0 |', include this exact table format in your response.\n"
                system_prompt += "4. You can reference tables with phrases like 'As shown in the table below:' or 'The data is presented in Table X:' before displaying the Markdown table.\n"
                system_prompt += "5. DO NOT convert tables to plain text - preserve the Markdown table structure (with pipes | and dashes ---) so they render properly.\n"
            
            # Add source information for citation
            if document_metadata and len(document_metadata) > 0:
                sources_text = []
                for meta in document_metadata:
                    source_parts = []
                    if meta.get('filename'):
                        # Clean filename (remove path, keep just name)
                        filename = meta['filename'].split('/')[-1].split('\\')[-1]
                        source_parts.append(filename)
                    if meta.get('author'):
                        source_parts.append(f"by {meta['author']}")
                    if meta.get('page'):
                        source_parts.append(f"page {meta['page']}")
                    
                    if source_parts:
                        sources_text.append(" ".join(source_parts))
                
                if sources_text:
                    system_prompt += f"\n\nSOURCE INFORMATION (MUST CITE IN RESPONSE):\nThe information above comes from the following source(s):\n" + "\n".join([f"- {source}" for source in sources_text])
                    system_prompt += "\n\nCRITICAL SOURCE CITATION RULES:\n"
                    system_prompt += "1. You MUST include these specific source names in your 'Sources' section at the end of your response.\n"
                    system_prompt += "2. Do NOT just say 'RAG documents' - list the actual filenames and page numbers.\n"
                    system_prompt += "3. ONLY cite sources that actually contain information you used in your response.\n"
                    system_prompt += "4. If a source was retrieved but you didn't use its content, do NOT cite it.\n"
                    system_prompt += "5. Be accurate - if you only used information from one source, only cite that one source."
            
            # ✅ Add image information if available (with captions and figure numbers)
            if associated_images and len(associated_images) > 0:
                total_images_available = len(associated_images)
                
                # Group images by figure number/page/filename so duplicate crops don't repeat instructions
                image_groups: "OrderedDict" = OrderedDict()
                for img in associated_images:
                    figure_number = (img.get('figure_number') or '').strip()
                    page = img.get('page')
                    filename = (img.get('filename') or '').strip()
                    filename_key = filename.lower()
                    
                    if figure_number:
                        key = (figure_number.lower(), page, filename_key)
                    else:
                        key = (img.get('image_url'), page, filename_key)
                    
                    if key not in image_groups:
                        image_groups[key] = {"image": img, "count": 1}
                    else:
                        image_groups[key]["count"] += 1
                
                unique_images_for_prompt = [group["image"] for group in image_groups.values()]
                images_info = []
                for idx, (key, group) in enumerate(list(image_groups.items())[:5], 1):  # Limit to top 5 unique images
                    img = group["image"]
                    duplicate_count = group["count"]
                    
                    img_desc = f"Image {idx} from {img.get('filename', 'document')}"
                    if img.get('page'):
                        img_desc += f" (page {img['page']})"
                    
                    # ✅ Include figure number and caption if available
                    if img.get('figure_number'):
                        img_desc += f" - FIGURE {img['figure_number']}"
                    if img.get('caption'):
                        caption_preview = img['caption'][:150] + "..." if len(img['caption']) > 150 else img['caption']
                        img_desc += f": {caption_preview}"
                    elif img.get('captions') and len(img.get('captions', [])) > 0:
                        # Fallback to first caption if caption field not available
                        first_caption = img['captions'][0]
                        caption_preview = first_caption[:150] + "..." if len(first_caption) > 150 else first_caption
                        img_desc += f": {caption_preview}"
                    
                    if img.get('ocr_text'):
                        ocr_preview = img['ocr_text'][:100] + "..." if len(img['ocr_text']) > 100 else img['ocr_text']
                        img_desc += f" - Also contains text: {ocr_preview}"
                    
                    if duplicate_count > 1:
                        extra_views = duplicate_count - 1
                        img_desc += f" (+{extra_views} additional view{'s' if extra_views > 1 else ''} of the same figure)"
                    
                    images_info.append(img_desc)
                
                system_prompt += f"\n\n📷 AVAILABLE IMAGES: The documents contain {total_images_available} diagram(s)/image(s) related to your query (grouped into {len(unique_images_for_prompt)} unique figure reference{'s' if len(unique_images_for_prompt) != 1 else ''}):\n"
                system_prompt += "\n".join([f"- {img_info}" for img_info in images_info])
                system_prompt += f"\n\n🎓 CRITICAL INSTRUCTIONS FOR IMAGE/DIAGRAM REFERENCING (Works for ALL domains - academic, health, business, technical, etc.):\n"
                system_prompt += f"1. When explaining concepts that relate to the images/diagrams, NATURALLY reference them in your response.\n"
                
                # ✅ Build figure reference examples from actual (deduplicated) images
                figure_refs = []
                for idx, (key, group) in enumerate(list(image_groups.items())[:5], 1):
                    img = group["image"]
                    if img.get('figure_number'):
                        figure_refs.append(f"'See FIGURE {img['figure_number']}' or 'As shown in FIGURE {img['figure_number']}'")
                
                primary_image = unique_images_for_prompt[0] if unique_images_for_prompt else {}
                secondary_image = unique_images_for_prompt[1] if len(unique_images_for_prompt) > 1 else None
                primary_figure = primary_image.get('figure_number', 'X') if primary_image else 'X'
                primary_page = primary_image.get('page', 'X') if primary_image else 'X'
                secondary_figure = secondary_image.get('figure_number') if secondary_image else None
                
                if figure_refs:
                    ref_examples = " or ".join(figure_refs[:3])  # Limit to 3 examples
                    system_prompt += (
                        f"2. Use FIGURE numbers when available: {ref_examples}. For academic/educational domains, use 'diagram' or 'FIGURE'; "
                        f"for health domains, use 'illustration' or 'visual'; for business, use 'chart' or 'graph'; for technical, use 'diagram' or 'visualization'. "
                        f"Use phrases like: 'As shown in Image 1 below...', 'Refer to FIGURE {primary_figure} on page {primary_page}...', "
                        f"'This figure provides visual reinforcement for the explanation.'\n"
                    )
                else:
                    system_prompt += (
                        f"2. Adapt your language to the domain: For academic/educational domains, use 'diagram'; for health domains, use 'illustration' or 'visual'; "
                        f"for business, use 'chart' or 'graph'; for technical, use 'diagram' or 'visualization'. "
                        f"Use phrases like: 'As shown in Image 1 below...', 'Refer to the visual on page {primary_page}...', 'This diagram supports the explanation.'\n"
                    )
                
                system_prompt += f"3. DO NOT just say 'images are available' - instead, actively reference them as if they were part of your explanation.\n"
                
                if secondary_image:
                    secondary_text = f"FIGURE {secondary_figure}" if secondary_figure else "the second visual"
                else:
                    secondary_text = "the second visual"
                
                system_prompt += (
                    f"4. For multiple images, differentiate them using their figure numbers or positions: "
                    f"'FIGURE {primary_figure} shows...' while '{secondary_text}' illustrates complementary detail. "
                    f"If multiple images share the same figure number, reference that figure once and note that additional views are available instead of repeating the same reference multiple times.\n"
                )
                system_prompt += f"5. Make the text flow naturally - images/visuals are supporting aids, so reference them contextually throughout your response.\n"
                system_prompt += f"6. When you mention a concept that an image illustrates, immediately reference that image (with its figure number if available) to reinforce the connection.\n"
                system_prompt += f"7. For non-academic domains, adapt the referencing style: business domains might reference 'charts' or 'graphs', health domains 'illustrations' or 'diagrams', technical domains 'schematics' or 'visualizations'.\n"
                system_prompt += f"8. IMPORTANT: DO NOT describe images in detail or repeat their content in your response. The frontend will display images separately with their captions. Just reference them briefly using their figure numbers (e.g., 'See FIGURE 1.8' or 'As shown in the diagram'). Do NOT embed image descriptions, study tips, or detailed image content in your markdown response - these are handled by the frontend.\n"
        else:
            system_prompt += "\n\nNote: No specific context documents are available. Please use your general knowledge to answer the question."
        
        # Add final STOP reminder right before user query (last thing model sees)
        system_prompt += "\n\n🛑 FINAL REMINDER: After completing the Sources section, STOP generating immediately. Do NOT write anything after Sources."
        
        # Build final prompt
        if model_type == "thinking":
            # Thinking model gets more detailed instructions
            prompt = f"""{system_prompt}

Human: {query}

Assistant: Let me think about this step by step. Based on the context and your question about {domain}, I'll provide a comprehensive response."""
        else:
            # Instruct model gets direct instructions
            prompt = f"""{system_prompt}

Human: {query}

Assistant:"""
        
        return prompt
    
    def _fallback_response(
        self, 
        query: str, 
        context: str, 
        domain: str, 
        emotion: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback response when models are not available.
        
        ✅ CONFIG-DRIVEN: Uses domain_config.yaml instead of hardcoded responses.
        """
        # ✅ Get fallback message from config
        response = config_loader.get_fallback_message(domain)
        
        return {
            "response": response,
            "model_used": "fallback",
            "domain": domain,
            "emotion": emotion,
            "context_used": False,
            "generation_time": 0.1,
            "tokens_generated": len(response.split()),
            "success": False,
            "fallback_reason": "Models not loaded"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            "meetara_models_enabled": self.config.use_meetara_models,
            "models_path": str(self.config.meetara_models_path),
            "loaded_models": list(self.models.keys()),
            "available_models": []
        }
        
        # Check available models
        models_path = self.config.meetara_models_path
        if models_path.exists():
            for model_file in models_path.glob("*.gguf"):
                info["available_models"].append(model_file.name)
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of GGUF models."""
        health = {
            "status": "healthy" if self.models else "unhealthy",
            "models_loaded": len(self.models),
            "instruct_available": "instruct" in self.models,
            "thinking_available": "thinking" in self.models,
            "models_path_exists": self.config.meetara_models_path.exists()
        }
        
        return health


# Global instance
meetara_gguf_processor = None

def get_meetara_gguf_processor() -> MeetaraGGUFProcessor:
    """Get or create global GGUF processor instance."""
    global meetara_gguf_processor
    if meetara_gguf_processor is None:
        meetara_gguf_processor = MeetaraGGUFProcessor()
    return meetara_gguf_processor
