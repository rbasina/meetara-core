"""
GGUF LLM Processor for Meetara Core
Uses custom fine-tuned GGUF models for optimal performance
Supports automatic download from Hugging Face with caching
"""

import os
import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding, LlamaDraftModel
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
import numpy.typing as npt
from typing import Any as TypingAny

# Check llama-cpp-python version for compatibility diagnostics
try:
    import llama_cpp
    LLAMA_CPP_VERSION = getattr(llama_cpp, '__version__', 'unknown')
    try:
        LLAMA_CPP_PATH = str(Path(llama_cpp.__file__).resolve()) if hasattr(llama_cpp, '__file__') else 'unknown'
    except:
        LLAMA_CPP_PATH = str(llama_cpp.__file__) if hasattr(llama_cpp, '__file__') else 'unknown'
except ImportError:
    LLAMA_CPP_VERSION = 'not installed'
    LLAMA_CPP_PATH = 'not installed'


class SafePromptLookupDecoding(LlamaDraftModel):
    """
    A safer wrapper around LlamaPromptLookupDecoding that handles edge cases.
    
    The original LlamaPromptLookupDecoding can cause broadcasting errors in llama-cpp-python
    when no n-gram matches are found (returns empty array). This wrapper ensures we always
    return a valid array that won't cause issues.
    """
    
    def __init__(self, max_ngram_size: int = 3, num_pred_tokens: int = 10, min_input_length: int = 50):
        self.inner = LlamaPromptLookupDecoding(
            max_ngram_size=max_ngram_size,
            num_pred_tokens=num_pred_tokens
        )
        self.min_input_length = min_input_length  # Don't try speculative decoding on short inputs
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens
    
    def __call__(
        self, input_ids: npt.NDArray[np.intc], /, **kwargs: TypingAny
    ) -> npt.NDArray[np.intc]:
        # Skip speculative decoding for short inputs (not enough context for n-gram matching)
        if len(input_ids) < self.min_input_length:
            return np.array([], dtype=np.intc)
        
        try:
            result = self.inner(input_ids, **kwargs)
            return result
        except Exception:
            # If anything goes wrong, return empty array (no speculation)
            return np.array([], dtype=np.intc)


from app.core.logger import agent_logger
from app.core.config import Settings
from app.core.domain_categorizer import get_domain_categorizer  # ✅ Optimized domain categorization
from app.core.config_loader import config_loader  # ✅ Use config for fallback messages


# ============================================================
# PRE-COMPILED REGEX PATTERNS (compiled once at module load)
# ============================================================
# These patterns are used for removing thinking text from LLM responses.
# Pre-compiling them provides ~10-15% performance improvement.

# Thinking block patterns (multi-line)
THINKING_BLOCK_PATTERNS = [
    re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<thinking>.*?</thinking>', re.DOTALL | re.IGNORECASE),
    re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[thinking\].*?\[/thinking\]', re.DOTALL | re.IGNORECASE),
    re.compile(r'\[internal\].*?\[/internal\]', re.DOTALL | re.IGNORECASE),
    re.compile(r'🧠\s*Thinking:.*?(?=\n\n|\*\*Quick|\Z)', re.DOTALL),
]

# Inline thinking patterns (single line, applied globally)
INLINE_THINKING_PATTERNS = [
    # Meta-commentary about user
    re.compile(r'The user\s+(?:is asking|wants|needs|seems|appears|might|could)[^.!?]*[.!?]', re.IGNORECASE),
    re.compile(r"The user's question is about[^.!?]*[.!?]\s*", re.IGNORECASE),
    # Self-referential planning
    re.compile(r'I (?:should|need to|will|must|can|could|would|might)\s+(?:present|provide|include|mention|focus|start|make sure|check|ensure|list|add|use)[^.!?]*[.!?]', re.IGNORECASE),
    re.compile(r"I'll (?:use|include|focus|list|mention|present|provide|start|make)[^.!?]*[.!?]\s*", re.IGNORECASE),
    # Context references
    re.compile(r'The context\s+(?:mentions|also mentions|includes|contains|shows|states|provides|suggests)[^.!?]*[.!?]', re.IGNORECASE),
    re.compile(r'(?:From|Based on|According to) the (?:provided |given )?context[^.!?]*[.!?]', re.IGNORECASE),
    # Section planning
    re.compile(r'(?:The|This|Each) (?:Quick Answer|Emotional|Physical|Social|section)\s+(?:should|has|must|needs|could|might)[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'Each section (?:must|should|has|needs)[^.!?]*[.!?]\s*', re.IGNORECASE),
    # Source planning
    re.compile(r'The (?:sources|Sources) (?:are|section|lists|need)[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'Sources need to be cited[^.!?]*[.!?]\s*', re.IGNORECASE),
    # Wait/Okay starters
    re.compile(r'Wait,?\s+the user[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'^(?:Okay|Ok|Alright),?\s+(?:so|let me|I)[^.!?]*[.!?]\s*', re.IGNORECASE | re.MULTILINE),
    # Let me/Now patterns
    re.compile(r'^Let me\s+(?:start|think|analyze|check|make sure|break)[^.!?]*[.!?]\s*', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^Now,?\s+(?:let me|I\'ll|I will|I need)[^.!?]*[.!?]\s*', re.IGNORECASE | re.MULTILINE),
    # Instruction compliance
    re.compile(r'Avoid any internal[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'Check for any markdown[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'Ensure the sections[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'Also,? ensure (?:the|that)[^.!?]*[.!?]\s*', re.IGNORECASE),
    # Planning language
    re.compile(r'(?:Tips could include|Common pitfalls|Expert recommendations could be)[^.!?]*[.!?]\s*', re.IGNORECASE),
    re.compile(r'\?\s*(?:Make the offer|This plan can help|Would you like me to)[^.!?]*[.!?]?\s*', re.IGNORECASE),
]

# Line-start thinking patterns (for line-by-line filtering)
LINE_START_THINKING_PATTERNS = [
    re.compile(r'^the user is asking', re.IGNORECASE),
    re.compile(r'^i should', re.IGNORECASE),
    re.compile(r'^i need to', re.IGNORECASE),
    re.compile(r'^the context', re.IGNORECASE),
    re.compile(r'^according to', re.IGNORECASE),
    re.compile(r'^from the provided', re.IGNORECASE),
    re.compile(r'^this section', re.IGNORECASE),
    re.compile(r'^maybe include', re.IGNORECASE),
    re.compile(r'^sources need', re.IGNORECASE),
    re.compile(r'^avoid any', re.IGNORECASE),
    re.compile(r'^check for', re.IGNORECASE),
    re.compile(r'^ensure the', re.IGNORECASE),
    re.compile(r'^wait,', re.IGNORECASE),
    re.compile(r'^okay,', re.IGNORECASE),
    re.compile(r'^so,', re.IGNORECASE),
    re.compile(r'^now,', re.IGNORECASE),
    re.compile(r"^i'll", re.IGNORECASE),
    re.compile(r'^the quick answer', re.IGNORECASE),
    re.compile(r'^the emotional', re.IGNORECASE),
    re.compile(r'^the physical', re.IGNORECASE),
    re.compile(r'^the social', re.IGNORECASE),
    re.compile(r'^each section', re.IGNORECASE),
]

# Placeholder patterns
PLACEHOLDER_PATTERNS = [
    re.compile(r'\[(?:Your|The|A|An|Write|Add|Include|Provide|Insert|Enter|Put|Fill)[^\]]{0,100}\]', re.IGNORECASE),
    re.compile(r'\[(?:Rich|Detailed|Specific|Relevant|Appropriate)[^\]]{0,100}\]', re.IGNORECASE),
    re.compile(r'\[(?:2-3|3-5|1-2|4-6)\s+(?:sentences?|points?|items?|examples?)[^\]]*\]', re.IGNORECASE),
]

# Source file pattern (for truncation after sources)
SOURCE_FILE_PATTERN = re.compile(r'[-•]\s*[^\n]+\.(?:pdf|doc|docx|txt)[^\n]*', re.IGNORECASE)


class MeetaraGGUFProcessor:
    """GGUF LLM processor using custom fine-tuned Meetara models."""
    
    def __init__(self):
        self.config = Settings()
        self.models = {}  # Will store loaded models
        self.model_paths = {}  # Will store paths for lazy loading
        self._model_cache = {}
        self.domain_categorizer = get_domain_categorizer()  # ✅ Use optimized categorizer
        
        # Log version info at initialization
        import sys
        python_exe = sys.executable
        agent_logger.info(f"📦 llama-cpp-python version: {LLAMA_CPP_VERSION}")
        agent_logger.info(f"   Installed at: {LLAMA_CPP_PATH}")
        agent_logger.info(f"   Python executable: {python_exe}")
        
        if "venv" in LLAMA_CPP_PATH.lower() or ".venv" in LLAMA_CPP_PATH.lower():
            agent_logger.info(f"   ✅ Using virtual environment version")
        elif "miniconda" in LLAMA_CPP_PATH.lower() or "conda" in LLAMA_CPP_PATH.lower():
            agent_logger.error(f"   ❌ PROBLEM: Using base conda environment (old version 0.3.9)!")
            agent_logger.error(f"   💡 Solution: Make sure venv is activated:")
            agent_logger.error(f"      .venv-meetara\\Scripts\\Activate.ps1  (PowerShell)")
            agent_logger.error(f"      OR use venv Python directly: .venv-meetara\\Scripts\\python.exe")
            if "venv" not in python_exe.lower() and ".venv" not in python_exe.lower():
                agent_logger.error(f"   ⚠️  Python executable is from conda, not venv!")
        
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
            model_path = self.model_paths[model_type]
            model_file = Path(model_path)
            
            agent_logger.info(f"🔄 Loading {model_type} model on first use: {model_path}")
            
            # Pre-load validation: Check if file exists and is readable
            if not model_file.exists():
                agent_logger.error(f"❌ Model file does not exist: {model_path}")
                return False
            
            file_size_mb = model_file.stat().st_size / (1024 * 1024)
            if file_size_mb < 100:  # GGUF files should be at least 100MB
                agent_logger.warning(f"⚠️ Model file seems unusually small ({file_size_mb:.1f} MB). May be corrupted.")
            
            agent_logger.info(f"   File size: {file_size_mb:.1f} MB")
            
            start_time = time.time()
            
            # ✅ SPEED OPTIMIZATIONS for llama-cpp-python
            # n_batch: Number of tokens to process in parallel (higher = faster, more memory)
            # n_gpu_layers: Offload layers to GPU if available (0 = CPU only)
            # flash_attn: Use Flash Attention if supported (faster attention computation)
            n_threads = os.cpu_count() or 4
            n_batch = 512  # Process 512 tokens at a time (default is 512, can go higher if RAM allows)
            
            # Check if GPU is available (CUDA)
            n_gpu_layers = 0  # Default: CPU only
            try:
                import torch
                if torch.cuda.is_available():
                    n_gpu_layers = -1  # Offload ALL layers to GPU for maximum speed
                    agent_logger.info(f"🚀 GPU detected! Offloading all layers to CUDA for faster inference")
            except ImportError:
                pass  # No torch, stay on CPU
            
            # ✅ SPECULATIVE DECODING - Prompt Lookup Decoding
            # Uses n-gram matching from the prompt/context to predict multiple tokens at once
            # Perfect for RAG scenarios where context contains similar patterns to the expected output
            # NOTE: Using SafePromptLookupDecoding wrapper to handle edge cases in llama-cpp-python
            draft_model = None
            if self.config.enable_speculative_decoding:
                draft_model = SafePromptLookupDecoding(
                    max_ngram_size=self.config.speculative_max_ngram_size,
                    num_pred_tokens=self.config.speculative_num_pred_tokens,
                    min_input_length=100  # Need at least 100 tokens for meaningful n-gram matching
                )
                agent_logger.info(f"⚡ Speculative decoding enabled: {self.config.speculative_max_ngram_size}-gram, {self.config.speculative_num_pred_tokens} tokens lookahead (safe mode)")
            
            # Try loading with speculative decoding first
            try:
                self.models[model_type] = Llama(
                    model_path=str(self.model_paths[model_type]),
                    n_ctx=self.config.llm_context_length,
                    n_threads=n_threads,
                    n_batch=n_batch,  # ✅ Parallel token processing
                    n_gpu_layers=n_gpu_layers,  # ✅ GPU acceleration if available
                    draft_model=draft_model,  # ✅ Speculative decoding
                    verbose=False
                )
                
                # ✅ Check if model has baked-in chat template (from enhanced GGUF conversion)
                try:
                    model_metadata = getattr(self.models[model_type], 'metadata', {})
                    chat_template = model_metadata.get('tokenizer.chat_template', '')
                    if chat_template:
                        # Check if it's the enhanced 5-section emoji template
                        has_emoji_template = any(emoji in chat_template for emoji in ['🎯', '📊', '⚡', '💡', '🤔'])
                        if has_emoji_template:
                            agent_logger.info(f"📋 Model has baked-in enhanced chat template with 16 intelligence layers (5-section emoji format)")
                            agent_logger.info(f"   🧠 Intelligence layers active: Refusal, Emotion, Intent, Tone, Domain, Safety, etc.")
                            agent_logger.info(f"   ⚠️ Overriding Layer 16 (emoji format) with domain-specific sections")
                            agent_logger.debug(f"   Template preview: {chat_template[:200]}...")
                        else:
                            agent_logger.info(f"📋 Model has baked-in chat template (custom format)")
                            agent_logger.debug(f"   Template preview: {chat_template[:200]}...")
                    else:
                        agent_logger.info(f"📋 Model does not have baked-in chat template (using prompt-only)")
                except Exception as meta_error:
                    agent_logger.debug(f"   Could not read model metadata: {meta_error}")
                
                load_time = time.time() - start_time
                device_info = "GPU (CUDA)" if n_gpu_layers != 0 else f"CPU ({n_threads} threads)"
                spec_info = " + Speculative Decoding" if draft_model else ""
                agent_logger.info(f"✅ {model_type.capitalize()} model loaded in {load_time:.1f}s on {device_info}{spec_info}")
                return True
            except Exception as load_error:
                error_str = str(load_error).lower()
                
                # Check if it's a GGUF parsing error (common with new/updated models)
                if "unexpected char" in error_str or "#" in error_str or "gguf" in error_str or "parse" in error_str:
                    agent_logger.warning(f"⚠️ GGUF parsing error detected: {load_error}")
                    
                    # Strategy 1: Try without speculative decoding
                    agent_logger.info(f"🔄 Strategy 1: Attempting to load without speculative decoding...")
                    try:
                        self.models[model_type] = Llama(
                            model_path=str(self.model_paths[model_type]),
                            n_ctx=self.config.llm_context_length,
                            n_threads=n_threads,
                            n_batch=n_batch,
                            n_gpu_layers=n_gpu_layers,
                            # No draft_model
                            verbose=False
                        )
                        load_time = time.time() - start_time
                        device_info = "GPU (CUDA)" if n_gpu_layers != 0 else f"CPU ({n_threads} threads)"
                        agent_logger.info(f"✅ {model_type.capitalize()} model loaded in {load_time:.1f}s on {device_info} (without speculative decoding)")
                        agent_logger.warning(f"⚠️ Speculative decoding disabled for this model due to GGUF compatibility")
                        return True
                    except Exception as retry1_error:
                        agent_logger.warning(f"⚠️ Strategy 1 failed: {retry1_error}")
                        
                        # Strategy 2: Try with use_mmap=False (sometimes fixes metadata reading issues)
                        agent_logger.info(f"🔄 Strategy 2: Attempting to load with use_mmap=False...")
                        try:
                            self.models[model_type] = Llama(
                                model_path=str(self.model_paths[model_type]),
                                n_ctx=self.config.llm_context_length,
                                n_threads=n_threads,
                                n_batch=n_batch,
                                n_gpu_layers=n_gpu_layers,
                                use_mmap=False,  # Disable memory mapping - may help with corrupted metadata
                                verbose=False
                            )
                            load_time = time.time() - start_time
                            device_info = "GPU (CUDA)" if n_gpu_layers != 0 else f"CPU ({n_threads} threads)"
                            agent_logger.info(f"✅ {model_type.capitalize()} model loaded in {load_time:.1f}s on {device_info} (use_mmap=False)")
                            agent_logger.warning(f"⚠️ Memory mapping disabled for this model due to GGUF compatibility")
                            return True
                        except Exception as retry2_error:
                            agent_logger.warning(f"⚠️ Strategy 2 failed: {retry2_error}")
                            
                            # Strategy 3: Try minimal parameters (most compatible)
                            agent_logger.info(f"🔄 Strategy 3: Attempting with minimal parameters (max compatibility)...")
                            try:
                                self.models[model_type] = Llama(
                                    model_path=str(self.model_paths[model_type]),
                                    n_ctx=2048,  # Smaller context
                                    n_threads=n_threads,
                                    # Minimal parameters only
                                    verbose=False
                                )
                                load_time = time.time() - start_time
                                agent_logger.info(f"✅ {model_type.capitalize()} model loaded in {load_time:.1f}s (minimal parameters)")
                                agent_logger.warning(f"⚠️ Using minimal parameters due to GGUF compatibility issues")
                                return True
                            except Exception as retry3_error:
                                agent_logger.error(f"❌ All retry strategies failed. Last error: {retry3_error}")
                                agent_logger.error(f"   Original error: {load_error}")
                                raise load_error  # Raise original error
                else:
                    # Re-raise if it's a different error
                    raise
                    
        except Exception as e:
            error_str = str(e)
            model_path = self.model_paths.get(model_type, "unknown")
            agent_logger.error(f"❌ Failed to load {model_type} model: {error_str}")
            agent_logger.error(f"   Model path: {model_path}")
            agent_logger.error(f"   Model exists: {Path(model_path).exists() if isinstance(model_path, (str, Path)) else 'N/A'}")
            
            # Provide helpful suggestions
            if "unexpected char" in error_str.lower() or "#" in error_str.lower():
                agent_logger.error(f"💡 This is a GGUF format compatibility issue (NOT a model problem).")
                agent_logger.error(f"   Current llama-cpp-python version: {LLAMA_CPP_VERSION}")
                agent_logger.error(f"   Installed at: {LLAMA_CPP_PATH}")
                
                # Check if they're using latest llama.cpp from GitHub
                local_llama_cpp_path = Path(r"C:\Users\rames\Documents\github\meetara\llama.cpp")
                if local_llama_cpp_path.exists():
                    agent_logger.error(f"   🔍 Detected: You're creating GGUF files with latest llama.cpp from GitHub")
                    agent_logger.error(f"   ⚠️  WARNING: Your GGUF files use the LATEST format, but llama-cpp-python {LLAMA_CPP_VERSION} may not support it!")
                    
                if LLAMA_CPP_VERSION.startswith('0.3.9') or LLAMA_CPP_VERSION.startswith('0.2.'):
                    agent_logger.error(f"   ⚠️  WARNING: Version {LLAMA_CPP_VERSION} is too old for newer GGUF formats!")
                elif LLAMA_CPP_VERSION.startswith('0.3.16'):
                    agent_logger.error(f"   ⚠️  WARNING: Version 0.3.16 may not support the latest GGUF format from GitHub llama.cpp!")
                
                agent_logger.error(f"   Possible causes:")
                agent_logger.error(f"   1. Your GGUF was created with LATEST llama.cpp (GitHub), but llama-cpp-python {LLAMA_CPP_VERSION} uses older llama.cpp")
                agent_logger.error(f"   2. GGUF metadata format mismatch (the '#' character at byte 82920 suggests metadata parsing issue)")
                agent_logger.error(f"   3. File may be corrupted during download (less likely)")
                agent_logger.error(f"   💡 Solutions to try:")
                agent_logger.error(f"   - Build llama-cpp-python from source with your local llama.cpp:")
                agent_logger.error(f"     git clone https://github.com/abetlen/llama-cpp-python.git")
                agent_logger.error(f"     cd llama-cpp-python")
                agent_logger.error(f"     # Point to your local llama.cpp:")
                agent_logger.error(f"     git clone C:/Users/rames/Documents/github/meetara/llama.cpp vendor/llama.cpp")
                agent_logger.error(f"     pip install -e .")
                agent_logger.error(f"   - OR update llama-cpp-python to latest: pip install --upgrade llama-cpp-python")
                agent_logger.error(f"   - OR use an older llama.cpp commit when converting GGUF (compatible with llama-cpp-python {LLAMA_CPP_VERSION})")
                agent_logger.error(f"   - Make sure you're using the correct Python environment (check if venv is activated)")
            
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
            
            # ✅ SPEED OPTIMIZATION: Lower temperature = faster, more deterministic responses
            # Higher temperature causes more sampling iterations
            temperature = min(self.config.llm_temperature, 0.7)  # Cap at 0.7 for speed
            
            # ✅ LLM generation with retry for speculative decoding errors
            try:
                response = self.models[model_type](
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config.local_llm_top_p,
                    top_k=self.config.local_llm_top_k,
                    # Note: Removed "**Sources**" from stop tokens - we WANT the model to generate Sources section
                    # Stop tokens to catch common continuation patterns after Sources
                    stop=["</s>", "<|im_end|>", "\n\nHuman:", "\n\nAssistant:", "(End of response)", "Final output:", "🛑", "STOP",
                          "\n\n\n**", "\n\n**Conclusion", "\n\n**Summary"],  # Stop if model tries to add new sections after Sources
                    echo=False
                )
            except Exception as gen_error:
                # Check if this is a speculative decoding broadcasting error
                error_str = str(gen_error)
                if "broadcast" in error_str.lower() or "shape" in error_str.lower():
                    agent_logger.warning(f"⚠️ Speculative decoding error detected: {error_str[:100]}...")
                    agent_logger.info("🔄 Reloading model WITHOUT speculative decoding and retrying...")
                    
                    # Reload the model without speculative decoding
                    if model_type in self.models:
                        del self.models[model_type]
                    
                    # Temporarily disable speculative decoding
                    original_spec_setting = self.config.enable_speculative_decoding
                    self.config.enable_speculative_decoding = False
                    
                    # Reload model
                    if not self._load_model_lazy(model_type):
                        self.config.enable_speculative_decoding = original_spec_setting
                        raise gen_error  # Re-raise if reload fails
                    
                    # Retry generation
                    response = self.models[model_type](
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=self.config.local_llm_top_p,
                        top_k=self.config.local_llm_top_k,
                        stop=["</s>", "<|im_end|>", "\n\nHuman:", "\n\nAssistant:", "(End of response)", "Final output:", "🛑", "STOP"],
                        echo=False
                    )
                    agent_logger.info("✅ Retry successful without speculative decoding")
                    # Keep speculative decoding disabled for future calls
                else:
                    raise  # Re-raise non-speculative-decoding errors
            
            generation_time = time.time() - start_time
            
            # Extract response text
            response_text = response["choices"][0]["text"].strip()
            
            # ✅ DEBUG: Log raw response before post-processing (first 1000 chars)
            agent_logger.debug(f"📄 Raw model response (first 1000 chars):\n{response_text[:1000]}...")
            agent_logger.debug(f"📄 Full raw response length: {len(response_text)} chars")
            
            # ============================================================
            # CRITICAL: REMOVE MODEL THINKING/REASONING (Must happen FIRST)
            # Uses pre-compiled patterns from module level for ~15% speed boost
            # ============================================================
            
            # Remove XML-style thinking blocks using pre-compiled patterns
            for pattern in THINKING_BLOCK_PATTERNS:
                response_text = pattern.sub('', response_text)
            
            # AGGRESSIVE: Remove conversational reasoning throughout entire response
            # Uses pre-compiled patterns for ~15% speed improvement
            for _ in range(10):  # Reduced iterations since patterns are more efficient
                original_text = response_text
                
                # Apply all inline thinking patterns using pre-compiled regexes
                for pattern in INLINE_THINKING_PATTERNS:
                    response_text = pattern.sub('', response_text)
                
                # If no changes were made, we're done
                if response_text == original_text:
                    break
            
            # Remove multi-paragraph thinking blocks (more aggressive)
            # First, normalize line breaks to preserve bullet point lists
            # Group consecutive lines starting with bullets/dashes/numbers into single paragraphs
            lines = response_text.split('\n')
            normalized_lines = []
            current_bullet_group = []
            
            for line in lines:
                stripped = line.strip()
                # Check if line is a bullet point or list item
                is_bullet = (stripped.startswith(('•', '-', '*')) or 
                            (stripped and stripped[0].isdigit() and '.' in stripped[:3]))
                
                if is_bullet:
                    # If we were collecting a bullet group, add previous content first
                    if current_bullet_group:
                        normalized_lines.append('\n'.join(current_bullet_group))
                        current_bullet_group = []
                    current_bullet_group.append(line)
                elif current_bullet_group:
                    # Continue collecting if line is continuation of bullet (indented or continues previous)
                    if (stripped and not stripped.startswith(('**', '#')) and 
                        (line.startswith(' ') or len(stripped) < 50)):  # Likely continuation
                        current_bullet_group.append(line)
                    else:
                        # End of bullet group
                        normalized_lines.append('\n'.join(current_bullet_group))
                        current_bullet_group = []
                        normalized_lines.append(line)
                else:
                    normalized_lines.append(line)
            
            # Add any remaining bullet group
            if current_bullet_group:
                normalized_lines.append('\n'.join(current_bullet_group))
            
            # Now split into paragraphs
            paragraphs = '\n'.join(normalized_lines).split('\n\n')
            clean_paragraphs = []
            found_content = False
            
            thinking_indicators = [
                'the user', 'i need to', 'i should', 'let me', "i'll", 'first step',
                'second step', 'third step', 'for the core', 'for the effective',
                'for the emotional', 'for the physical', 'for the social', 'for the practical',
                'should include', 'should cover', 'should mention', 'might be',
                'could be', 'the query', 'this question', 'assessment and tracking',
                'i know that', 'i should mention', 'the context mentions', 'according to the provided',
                'the assistant could', 'the assistant must', 'sources should list',
                'the symptoms mentioned', 'the symptoms include', 'i should present',
                'this section might', 'this section needs', 'maybe include', 'perhaps mention',
                'the information shows', 'the data indicates', 'i need to find'
            ]
            
            for para in paragraphs:
                para_lower = para.lower().strip()
                if not para_lower:
                    clean_paragraphs.append(para)  # Keep empty paragraphs for spacing
                    continue
                
                # Check if paragraph is a section header (keep these)
                # Support both standard **headers** and enhanced template emoji headers (🎯, 📊, ⚡, 💡, 🤔)
                is_section_header = para_lower.startswith('**') and para_lower.endswith('**')
                is_emoji_header = any(emoji in para_lower[:10] for emoji in ['🎯', '📊', '⚡', '💡', '🤔'])
                if is_section_header or is_emoji_header:
                    clean_paragraphs.append(para)
                    found_content = True
                    continue
                
                # Check if paragraph contains content markers (keep these)
                # Support both standard format and enhanced template format
                # Note: Check for bullet points with or without space after (•text or • text)
                para_stripped = para_lower.strip()
                has_bullet_point = (para_stripped.startswith(('•', '-', '*')) or 
                                   (para_stripped and para_stripped[0].isdigit() and '.' in para_stripped[:3]))
                content_markers = ['**quick answer', '**', '# ', '## ', '- ', '• ', '•',  # Bullet point (with/without space)
                                 '1.', '2.', '3.', 'sources',
                                 '🎯', '📊', '⚡', '💡', '🤔',  # Enhanced template emoji markers
                                 'what this means', 'deeper understanding', 'practical steps', 
                                 'extra tips', 'thoughtful next question']  # Enhanced template section keywords
                has_content_marker = has_bullet_point or any(marker in para_lower for marker in content_markers)
                
                if has_content_marker:
                    clean_paragraphs.append(para)
                    found_content = True
                    continue
                
                # Check if paragraph is thinking/reasoning
                is_thinking = False
                
                # Check for thinking indicators at start or in first 150 chars
                for indicator in thinking_indicators:
                    if para_lower.startswith(indicator) or f' {indicator}' in para_lower[:150]:
                        is_thinking = True
                        break
                
                # Check for common thinking patterns
                if para_lower.startswith(('okay', 'so ', 'now ', 'next', 'also', 'for the', 'the user', 'i know', 'i should', 'let me start')):
                    is_thinking = True
                
                # Check if paragraph is mostly planning language (long paragraphs with thinking words)
                if len(para_lower) > 100:
                    thinking_word_count = sum(1 for word in ['should', 'could', 'might', 'need to', 'will', 'must'] if word in para_lower)
                    if thinking_word_count >= 3 and not has_content_marker:
                        is_thinking = True
                
                # Only keep non-thinking paragraphs, or thinking paragraphs if we haven't found real content yet
                if not is_thinking:
                    clean_paragraphs.append(para)
                    found_content = True
                elif not found_content:
                    # Keep thinking if it's before any real content (might be at very start)
                    clean_paragraphs.append(para)
            
            if clean_paragraphs:
                response_text = '\n\n'.join(clean_paragraphs)
            
            # ============================================================
            # REMOVE PLACEHOLDER BRACKETS AND TEMPLATE TEXT
            # Uses pre-compiled patterns for efficiency
            # ============================================================
            for pattern in PLACEHOLDER_PATTERNS:
                response_text = pattern.sub('', response_text)
            
            # Remove generic placeholder brackets like [text here] that weren't replaced
            # But preserve legitimate brackets like [1], [2], [a], [b] for citations
            response_text = re.sub(r'\[[^\]]{10,}\]', '', response_text)  # Remove brackets with 10+ chars inside
            
            # Remove empty bullet points that result from placeholder removal
            response_text = re.sub(r'^-\s*$', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^\d+\.\s*$', '', response_text, flags=re.MULTILINE)
            
            # Remove empty sections (section header with no content before next section)
            # Pattern: **Section Name** followed by only whitespace/newlines and then immediately another **Section**
            response_text = re.sub(
                r'\*\*[^*]+\*\*\s*\n{2,}(?=\*\*[^*]+\*\*)',
                '', response_text, flags=re.MULTILINE
            )
            
            # Remove placeholder text like "?" or incomplete placeholders
            response_text = re.sub(r'^\?\s*$', '', response_text, flags=re.MULTILINE)
            response_text = re.sub(r'^\?\s*Make the offer', '', response_text, flags=re.MULTILINE | re.IGNORECASE)
            
            # Remove thinking blocks that appear after sections (more aggressive)
            # Pattern: Section header followed by thinking text like "The symptoms mentioned..." or "I should present..."
            response_text = re.sub(
                r'(\*\*[^*]+\*\*)\s*\n+(?:The (?:symptoms|context|information|data) (?:mentioned|includes|contains|shows)|I should (?:present|include|mention)|This section (?:might|needs|should)|Maybe (?:include|mention)).*?(?=\n\n\*\*|\n\nSources|\Z)',
                r'\1\n\n', response_text, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
            )
            
            # Normalize multiple blank lines
            response_text = re.sub(r'\n{3,}', '\n\n', response_text)
            
            agent_logger.info(f"🧹 Post-processed response: removed thinking patterns and placeholders")
            
            # ✅ Minimal safety net: Only remove obvious duplicates (prevention should handle most cases)
            # Find first Sources section - response should end there
            # Check for multiple variations of Sources header
            first_sources_idx = -1
            for sources_pattern in ["**Sources**", "Sources\n", "Sources:", "**Source**"]:
                idx = response_text.find(sources_pattern)
                if idx > 0 and (first_sources_idx < 0 or idx < first_sources_idx):
                    first_sources_idx = idx
            
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
            
            # ============================================================
            # DEDUPLICATE SOURCES: Remove duplicate source entries
            # ============================================================
            # Find Sources section and deduplicate entries
            sources_match = re.search(r'(\*\*Sources\*\*|\*\*Source\*\*|Sources:)(.*?)(?=\n\n[A-Z]|\n\n\*\*|\Z)', response_text, flags=re.DOTALL | re.IGNORECASE)
            if sources_match:
                sources_header = sources_match.group(1)
                sources_content = sources_match.group(2)
                
                # Extract source lines (bullet points)
                source_lines = [line.strip() for line in sources_content.split('\n') if line.strip().startswith('-') or line.strip().startswith('•')]
                
                # Group by filename and collect page numbers
                file_pages = {}  # filename -> set of page numbers
                for source_line in source_lines:
                    # Normalize the source name (remove bullets, extra spaces)
                    normalized = re.sub(r'^[-•]\s*', '', source_line).strip()
                    
                    # Extract filename
                    filename_match = re.search(r'([^/\\]+\.(pdf|doc|docx|txt))', normalized, re.IGNORECASE)
                    if filename_match:
                        filename = filename_match.group(1)
                        
                        # Extract page numbers if present
                        page_match = re.search(r'page[s]?\s*(\d+(?:\s*[-,]\s*\d+)*)', normalized, re.IGNORECASE)
                        if page_match:
                            pages = page_match.group(1)
                        else:
                            pages = None
                        
                        if filename not in file_pages:
                            file_pages[filename] = set()
                        if pages:
                            file_pages[filename].add(pages)
                    else:
                        # Non-file source (e.g., "Based on general knowledge")
                        if normalized not in file_pages:
                            file_pages[normalized] = set()
                
                # Build unique sources list with combined page numbers
                unique_sources = []
                for filename, pages in file_pages.items():
                    if pages:
                        # Combine all page numbers
                        all_pages = sorted(set(p.strip() for p in ','.join(pages).replace('-', ',').split(',') if p.strip().isdigit()))
                        if all_pages:
                            unique_sources.append(f"- {filename} (pages {', '.join(all_pages)})")
                        else:
                            unique_sources.append(f"- {filename}")
                    else:
                        unique_sources.append(f"- {filename}")
                
                # Rebuild sources section with unique entries
                if len(unique_sources) < len(source_lines):
                    agent_logger.info(f"📚 Deduplicated sources: {len(source_lines)} → {len(unique_sources)} unique")
                new_sources = sources_header + '\n' + '\n'.join(unique_sources)
                response_text = response_text[:sources_match.start()] + new_sources
                # CRITICAL: Truncate EVERYTHING after Sources - no exceptions
                agent_logger.info(f"✂️ Truncated response at Sources section")
            
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
            
            # ============================================================
            # FINAL CLEANUP: Remove any remaining thinking that slipped through
            # ============================================================
            
            # CRITICAL: Find the last .pdf/.doc source and truncate everything after it
            # This catches cases where thinking text appears after sources without proper separation
            last_source_match = None
            for match in re.finditer(r'[-•]\s*[^\n]+\.(?:pdf|doc|docx|txt)[^\n]*', response_text, re.IGNORECASE):
                last_source_match = match
            
            if last_source_match:
                # Check if there's significant text after the last source
                text_after_sources = response_text[last_source_match.end():].strip()
                # If there's more than just whitespace/punctuation after sources, it's thinking text
                if text_after_sources and len(text_after_sources) > 10:
                    # Check if it starts with thinking patterns
                    thinking_starters = ['let me', 'i should', 'i need', 'wait,', 'the user', 'i\'ll', 'each section', 'the quick', 'under ', 'also,', 'so i', 'first,', 'then,']
                    if any(text_after_sources.lower().startswith(p) for p in thinking_starters):
                        response_text = response_text[:last_source_match.end()].strip()
                        agent_logger.warning(f"⚠️ Removed {len(text_after_sources)} chars of thinking text after sources")
            
            # These are last-resort patterns for text that escaped earlier passes
            final_cleanup_patterns = [
                # Sentences starting with meta-commentary
                r'^(?:The user|I should|I need to|Let me|I know that|The context|According to)[^.!?]*[.!?]\s*',
                # Sentences with planning language
                r'(?:Tips could include|Common pitfalls|Expert recommendations could be)[^.!?]*[.!?]\s*',
                # Self-referential instructions
                r'(?:Sources need to be cited|Avoid any internal|Check for any markdown|Ensure the sections)[^.!?]*[.!?]\s*',
                # Incomplete sentences with question marks followed by instructions
                r'\?\s*(?:Make the offer|This plan can help|Would you like me to)[^.!?]*[.!?]?\s*',
            ]
            for pattern in final_cleanup_patterns:
                response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove any lines that are ONLY thinking (no actual content)
            lines = response_text.split('\n')
            clean_lines = []
            # Use pre-compiled line-start patterns for efficiency
            for line in lines:
                line_lower = line.strip().lower()
                is_thinking_only = any(pattern.match(line_lower) for pattern in LINE_START_THINKING_PATTERNS)
                if not is_thinking_only:
                    clean_lines.append(line)
            response_text = '\n'.join(clean_lines)
            
            # Final normalization
            response_text = re.sub(r'\n{3,}', '\n\n', response_text).strip()
            
            # ✅ DEBUG: Check if response follows structure
            # Support both standard format and enhanced template format (with emojis)
            has_title = "**" in response_text and response_text.count("**") >= 2
            has_emoji_headers = any(emoji in response_text for emoji in ['🎯', '📊', '⚡', '💡', '🤔'])
            has_intro = len(response_text.split('\n')) > 3  # Has multiple lines (likely intro + body)
            has_sources = "**Sources**" in response_text or "Sources:" in response_text or "**Source**" in response_text
            
            # Log which format was detected
            if has_emoji_headers:
                agent_logger.info(f"✅ Detected enhanced template format (with emoji headers)")
            elif has_title:
                agent_logger.info(f"✅ Detected standard format (with ** headers)")
            else:
                agent_logger.warning(f"⚠️ No clear structure format detected - response may need formatting")
            
            # ✅ Check if response uses domain-specific section headers OR enhanced template format
            section_headers = self._get_domain_category_sections(domain)
            used_sections = []
            
            # Check for enhanced template format sections (with emojis)
            enhanced_template_sections = [
                ('🎯', 'What This Means for You', 'Direct Answer'),
                ('📊', 'Deeper Understanding', 'Key Details'),
                ('⚡', 'Practical Steps', 'You Can Take'),
                ('💡', 'Extra Tips', 'Warnings & Insights'),
                ('🤔', 'Thoughtful Next Question', 'for You')
            ]
            
            for emoji, section_name, keywords in enhanced_template_sections:
                if emoji in response_text or section_name.lower() in response_text.lower() or keywords.lower() in response_text.lower():
                    used_sections.append(f"{emoji} {section_name}")
            
            # Also check for domain-specific section headers (standard format)
            for section in section_headers:
                # Check if section header (without **) appears in response
                # Look for both with and without bold markers
                section_text = section.replace("**", "").strip()
                section_with_bold = section.strip()
                
                # Check multiple patterns: exact match, with bold, partial match (key words)
                if (section_text.lower() in response_text.lower() or 
                    section_with_bold.lower() in response_text.lower()):
                    if section_text not in [s.replace('🎯 ', '').replace('📊 ', '').replace('⚡ ', '').replace('💡 ', '').replace('🤔 ', '') for s in used_sections]:
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
- Based on general knowledge

INSTRUCTIONS:
- Use the structure shown in the example above
- Replace "[Your Title About the Topic]" with an actual title
- Replace "[2-3 sentences...]" with actual introduction text
- Select 2-3 section headers from the available list and use them exactly (with **bold**)
- Write 3-5 bullet points under each section header you choose
- Write a conclusion summarizing key takeaways
- List sources in **Sources** section using format: - DocumentName.pdf (page X). Always include page numbers in parentheses when available

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
            # ✅ DOMAIN-SPECIFIC SECTIONS: Use domain-appropriate section headers
            # The domain_categorizer provides domain-specific sections (e.g., academic_tutoring gets
            # "Core Concepts & Fundamentals", "Effective Learning Strategies", etc.)
            # 
            # 🧠 ENHANCED GGUF MODEL INTELLIGENCE (16 Layers - Working in Background):
            # The enhanced GGUF models have a baked-in chat template with 16 intelligence layers that
            # automatically detect and enhance responses with:
            # - Layer 1: 🚨 Refusal Check (safety-first harmful request detection)
            # - Layer 2: 🧩 Contextual Patterns (multi-word phrase disambiguation)
            # - Layer 3: 📊 N-gram Patterns (bigram/trigram detection)
            # - Layer 4: 🔗 Semantic Clusters (related keyword groups)
            # - Layer 5: 👤 Entity Patterns (personal context, time-sensitive)
            # - Layer 6: 🎯 Intent Signals (learn, fix, decide, create, validate)
            # - Layer 7: 💙 Emotional Intelligence (worried, frustrated, urgent, curious)
            # - Layer 8: 🎭 Tone Detection (casual, formal, technical)
            # - Layer 9: ❓ Question Type (yes/no, how-to, comparison)
            # - Layer 10: 📏 Response Length (concise/standard/detailed)
            # - Layer 11: 🎯 Domain Detection (weighted keyword scoring, 18 categories)
            # - Layer 12: ⚖️ Domain Priority (safety-critical domains win ties)
            # - Layer 13: 🔄 Context Awareness (follow-up detection)
            # - Layer 14: ⚠️ Safety Disclaimers (auto-adds warnings for healthcare, legal, crisis)
            # - Layer 15: 👋 Greeting/Closing (natural conversation flow)
            # - Layer 16: 📝 Structured Responses (5-section format with emoji headers)
            #
            # NOTE: Since we're using raw prompts (not chat completion), the template layers don't
            # automatically process, but the model was fine-tuned with these intelligence features, so
            # it may still exhibit empathetic responses, appropriate tone, and domain awareness.
            #
            # We override Layer 16's emoji format (🎯, 📊, ⚡, 💡, 🤔) with domain-specific sections,
            # while still benefiting from the other 15 layers' intelligence (emotion, tone, intent, etc.)
            # Only fallback to generic emoji template if no domain-specific sections available
            
            if section_headers and len(section_headers) >= 2:
                # Use domain-specific sections (like general_knowledge does)
                domain_sections_list = "\n".join([f"{header}" for header in section_headers])
                
                system_prompt = f"""You are me²TARA, an advanced AI assistant specialized in {domain_display}. {format_instruction}

🧠 INTELLIGENCE CONTEXT: Your enhanced model has 16 intelligence layers that automatically detect emotion, intent, tone, and domain. 
Use these intelligently: if you detect frustration, lead with empathy. If you detect a "fix" intent, provide systematic steps. 
If you detect a technical query, use precise technical language. These layers enhance your responses automatically.

⚠️ STRUCTURE REQUIREMENT: You MUST use the domain-specific section headers below, NOT the generic emoji format (🎯, 📊, ⚡, 💡, 🤔).

RESPONSE STRUCTURE - Follow this exact format:

AVAILABLE SECTION HEADERS FOR {domain_display_title} (select 2-3 and use them exactly with **bold**):
{domain_sections_list}

EXAMPLE OF COMPLETE RESPONSE FORMAT (using {domain_display} domain headers):
Start your response directly with the first domain-specific section header (NO introduction, NO "Quick Answer", start immediately with the section):

{section_headers[0] if len(section_headers) > 0 else "**Core Concepts & Fundamentals**"}
[Write 2-3 sentences directly answering the question - this is your direct answer]
- [Key point 1 with specific information]
- [Key point 2 with details]
- [Key point 3 with context]

{section_headers[1] if len(section_headers) > 1 else "**Practical Application & Practice**"}
- [First actionable step or strategy]
- [Second actionable step or strategy]
- [Third actionable step or strategy]

{section_headers[2] if len(section_headers) > 2 else "**Additional Insights**"}
- [Additional tip or insight]
- [Important consideration]
- [Expert recommendation]

**Sources**
- [Format: - DocumentName.pdf (page X). List all sources used, one per line with bullet point and page numbers in parentheses]
- Example: - APBiology-OP.pdf (page 1232)
- Example: - Chemistry 2e.pdf (page 1013)
- [If no RAG documents were used, write: - Based on general knowledge]

INSTRUCTIONS:
- Start your response IMMEDIATELY with the first section header (NO introduction, NO "Quick Answer" section)
- Select 2-3 section headers from the available list and use them exactly (with **bold** formatting)
- Write 3-5 bullet points under each section header you choose
- The first section should directly answer the question in 2-3 sentences, then provide bullet points
- Include **Sources** section at the end with document names if RAG was used

🛑 STOP IMMEDIATELY AFTER SOURCES - THIS IS CRITICAL:
- After completing the Sources bullet points, STOP generating IMMEDIATELY - your response ENDS here
- The response ENDS at Sources - no repetition, no additional sections, no extra text
- Do NOT generate any text, explanations, conclusions, or summaries after Sources
- Your last line should be the last source bullet point (e.g., "- DocumentName.pdf (page X)") - that's where you STOP completely
- After Sources, there should be NOTHING - not even blank lines or punctuation

CRITICAL RULES:
- ❌ ABSOLUTELY NEVER use <think>, <reasoning>, or ANY internal thinking tags
- ❌ ABSOLUTELY NEVER start with meta-commentary like "Okay, I need to..." or "Let me think..." or "The user is asking..."
- ❌ ABSOLUTELY NEVER show your reasoning process or thought patterns
- ❌ ABSOLUTELY NEVER write "From the provided context" or "I should..." or planning language
- ❌ ABSOLUTELY NEVER use emoji headers (🎯, 📊, ⚡, 💡, 🤔) - use the domain-specific headers above instead
- ❌ ABSOLUTELY NEVER start with "Quick Answer" - start directly with the domain-specific section header
- ❌ ABSOLUTELY NEVER repeat any section after Sources - STOP at Sources
- ❌ ABSOLUTELY NEVER add text after Sources - the response ENDS at Sources, nothing after it
- ❌ Replace ALL placeholders with actual content - do NOT copy text like "[Your Title]" or "[2-3 sentences]"
- ✅ Start your response IMMEDIATELY with {section_headers[0] if len(section_headers) > 0 else "the first section header"}
- ✅ Use ONLY the domain-specific section headers shown above (with **bold** formatting), NOT emoji headers
- ✅ Do NOT add numbers or labels to section headers (just use the headers as-is)
- ✅ Generate the response ONCE - do NOT repeat sections
- ✅ STOP IMMEDIATELY after Sources section - your last character should be after the last source bullet point
- {structure_note}"""
            else:
                # Fallback to generic emoji template if no domain-specific sections available
                system_prompt = f"""You are me²TARA, an advanced AI assistant specialized in {domain_display}.

YOUR RESPONSE MUST FOLLOW THIS EXACT 5-SECTION FORMAT:

🎯 What This Means for You
[Write 2-3 sentences directly answering the question - this is your direct answer]

📊 Deeper Understanding & Key Details
- [Key point 1 with specific data or insights]
- [Key point 2 with research findings or expert knowledge]
- [Key point 3 with important context]

⚡ Practical Steps You Can Take
1. [First actionable step]
2. [Second actionable step]
3. [Third actionable step or follow-up]

💡 Extra Tips, Warnings & Insights
- [Practical tip or recommendation]
- [Common pitfall to avoid]
- [Expert advice or important warning]

🤔 Thoughtful Next Question for You
[Ask a relevant follow-up question to help the user think deeper about the topic]

**Sources**
- [Format: - DocumentName.pdf (page X). List all sources used, one per line with bullet point and page numbers in parentheses]
- Example: - APBiology-OP.pdf (page 1232)
- Example: - Chemistry 2e.pdf (page 1013)
- [If no RAG documents were used, write: - Based on general knowledge]

CRITICAL RULES:
1. Your response MUST start with "🎯 What This Means for You" - NOTHING before it
2. ❌ NEVER write thinking/reasoning blocks, meta-commentary, or planning language
3. ❌ NEVER write "From the provided context" or "The user is asking..." or "I should..."
4. ✅ Use the EXACT emoji headers shown above (🎯, 📊, ⚡, 💡, 🤔)
5. ✅ Write real, helpful content in each section - bullet points with actual information
6. ✅ Include **Sources** section at the end with document names if RAG was used
7. ✅ STOP immediately after Sources section - nothing more
8. {structure_note}"""
        
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
                    system_prompt += "1. You MUST include these specific source names in your '**Sources**' section at the end of your response.\n"
                    system_prompt += "2. Use EXACTLY this format for each source (one per line with bullet point):\n"
                    system_prompt += "   - DocumentName.pdf (page X)\n"
                    system_prompt += "   Example: - APBiology-OP.pdf (page 1232)\n"
                    system_prompt += "   Example: - Chemistry 2e.pdf (page 1013)\n"
                    system_prompt += "3. Use 'page' (singular) for single page, 'pages' (plural) only if multiple page numbers are provided.\n"
                    system_prompt += "4. ALWAYS include page numbers in parentheses format: (page X) or (pages X, Y)\n"
                    system_prompt += "5. ONLY cite sources that actually contain information you used in your response.\n"
                    system_prompt += "6. If a source was retrieved but you didn't use its content, do NOT cite it.\n"
                    system_prompt += "7. Be accurate - if you only used information from one source, only cite that one source."
            
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
        system_prompt += "\n\n🛑🛑🛑 CRITICAL FINAL REMINDER - READ THIS CAREFULLY: 🛑🛑🛑"
        system_prompt += "\n1. Your response MUST end with the Sources section - NOTHING after it."
        system_prompt += "\n2. After writing the last source bullet point (e.g., '- DocumentName.pdf (page 123)'), STOP generating IMMEDIATELY."
        system_prompt += "\n3. Do NOT add any conclusions, summaries, explanations, or additional text after Sources."
        system_prompt += "\n4. Do NOT start a new section or paragraph after Sources."
        system_prompt += "\n5. Your response ENDS at Sources - when you finish Sources, you are DONE."
        system_prompt += "\n6. If you think you need to add more, DON'T - the Sources section is your final section."
        
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
