"""
Configuration settings for Meetara Core backend.
"""
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Security
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:2025"],
        env="CORS_ORIGINS"
    )
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Vector Store Configuration
    vectorstore_path: Path = Field(
        default=Path("vectorstore"),
        env="VECTORSTORE_PATH"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")  # Smaller chunks for precise medical info
    chunk_overlap: int = Field(default=100, env="CHUNK_OVERLAP")  # Smaller overlap for better context
    
    # LLM Configuration (Offline)
    llm_model_path: Optional[Path] = Field(
        default=None,
        env="LLM_MODEL_PATH"
    )
    llm_context_length: int = Field(default=4096, env="LLM_CONTEXT_LENGTH")
    llm_temperature: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # Local LLM Configuration for RAG
    use_local_llm: bool = Field(default=True, env="USE_LOCAL_LLM")
    local_llm_model: str = Field(
        default="tiiuae/falcon-7b-instruct",
        env="LOCAL_LLM_MODEL"
    )
    
    # Custom Meetara GGUF Models
    # Note: Meetara fine-tuned models are quantized to GGUF format for efficient inference
    # Available models: https://huggingface.co/meetara-lab/models
    # - meetara-qwen3-1.7b-gguf (1.2 GB) - Fast, general purpose
    # - meetara-qwen3-4b-instruct-gguf (2.8 GB) - Instruction following, coding
    # - meetara-qwen3-4b-thinking-gguf (2.8 GB) - Deep reasoning, safety-critical
    # - meetara-qwen3-8b-gguf (5.5 GB) - Most capable
    
    # Default model to use (can be changed at runtime via API)
    # Options: meetara-1.7b, meetara-4b-instruct, meetara-4b-thinking, meetara-8b
    default_model: str = Field(
        default="meetara-1.7b",
        env="DEFAULT_MODEL"
    )
    
    # Auto-select model based on domain tier (if True, ignores default_model for domain queries)
    auto_select_model: bool = Field(
        default=True,
        env="AUTO_SELECT_MODEL"
    )
    
    # Hugging Face Model Configuration (for automatic download)
    # These are loaded from model_config.yaml, but can be overridden via env vars
    meetara_hf_model_id: Optional[str] = Field(
        default="meetara-lab/meetara-qwen3-1.7b-gguf",
        env="MEETARA_HF_MODEL_ID"
    )
    meetara_hf_model_file: str = Field(
        default="meetara-qwen3-1.7b-Q4_K_M.gguf",
        env="MEETARA_HF_MODEL_FILE"
    )
    
    # Local Model Paths (fallback if HF model ID not set)
    meetara_models_path: Path = Field(
        default=Path("models/gguf"),
        env="MEETARA_MODELS_PATH"
    )
    meetara_instruct_model: str = Field(
        default="meetara-qwen3-1.7b-Q4_K_M.gguf",
        env="MEETARA_INSTRUCT_MODEL"
    )
    meetara_thinking_model: str = Field(
        default="meetara-qwen3-4b-thinking.Q4_K_M.gguf",
        env="MEETARA_THINKING_MODEL"
    )
    use_meetara_models: bool = Field(default=True, env="USE_MEETARA_MODELS")
    
    # Model Cache Configuration (for HF downloads)
    # Uses standard Hugging Face cache location: ~/.cache/huggingface/hub
    # On Windows: C:\Users\<username>\.cache\huggingface\hub
    meetara_model_cache_dir: Optional[Path] = Field(
        default=None,  # None = use default HF cache location
        env="MEETARA_MODEL_CACHE_DIR"
    )
    local_llm_device: str = Field(default="cpu", env="LOCAL_LLM_DEVICE")  # "cpu" or "cuda"
    local_llm_max_length: int = Field(default=640, env="LOCAL_LLM_MAX_LENGTH")  # Tuned for faster responses while keeping full structure
    local_llm_do_sample: bool = Field(default=True, env="LOCAL_LLM_DO_SAMPLE")
    local_llm_top_p: float = Field(default=0.9, env="LOCAL_LLM_TOP_P")
    local_llm_top_k: int = Field(default=50, env="LOCAL_LLM_TOP_K")
    
    # Speculative Decoding Configuration
    # Enables faster inference by predicting multiple tokens at once
    # Uses prompt lookup decoding (n-gram matching) - perfect for RAG scenarios
    # NOTE: Disabled by default due to llama-cpp-python v0.3.x broadcasting bug
    # When no n-gram matches are found, it can cause "could not broadcast input array" errors
    # Enable only if you have llama-cpp-python >= 0.4.x or want to test
    enable_speculative_decoding: bool = Field(
        default=False,  # Disabled by default - see note above
        env="ENABLE_SPECULATIVE_DECODING"
    )
    speculative_max_ngram_size: int = Field(
        default=3,  # Look for 3-gram matches in context
        env="SPECULATIVE_MAX_NGRAM_SIZE"
    )
    speculative_num_pred_tokens: int = Field(
        default=10,  # Predict up to 10 tokens at a time
        env="SPECULATIVE_NUM_PRED_TOKENS"
    )
    
    # Speech Processing
    stt_model: str = Field(default="base", env="STT_MODEL")  # faster-whisper model
    tts_voice: str = Field(default="en-US-JennyNeural", env="TTS_VOICE")
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    
    # Emotion Detection
    emotion_models_path: Path = Field(
        default=Path("models/emotion"),
        env="EMOTION_MODELS_PATH"
    )
    face_detection_confidence: float = Field(default=0.5, env="FACE_DETECTION_CONFIDENCE")
    
    # Domain Configuration
    default_domains: List[str] = Field(
        default=[
            "academic_tutoring", "academic_tutoring_research", "aeronautics", "aerospace_engineering",
            "agriculture", "artificial_intelligence", "data_science", "machine_learning",
            "art_appreciation", "automobile", "career_guidance", "chronic_conditions",
            "communication", "conflict_resolution", "consulting", "content_creation",
            "creative_writing", "crisis_management", "customer_service", "cybersecurity",
            "data_analysis", "decision_making", "design_thinking", "digital_literacy",
            "disaster_preparedness", "education", "educational_technology", "emergency_care",
            "emergency_response", "engineering", "entrepreneurship", "exam_preparation",
            "financial", "financial_planning", "fitness_healthcare", "general_health",
            "home_management", "hr_management", "insurance", "language_learning_education",
            "language_learning_professional", "legal", "legal_assistance", "legal_business",
            "life_coaching", "manufacturing", "marketing", "medication_management",
            "mental_health", "music", "mythology", "nutrition", "operations",
            "parenting", "personal_assistant", "photography", "planning",
            "preventive_care", "programming", "project_management", "psychology",
            "real_estate", "relationships", "remote_work", "research",
            "research_assistance", "safety_security", "sales", "scientific_research",
            "senior_health", "shopping", "skill_development", "sleep",
            "social_media", "social_media_management", "social_support", "software_development",
            "space_technology", "spiritual", "sports_recreation", "storytelling",
            "stress_management", "study_techniques", "teaching", "team_leadership",
            "tech_support", "time_management", "transportation", "travel_tourism",
            "women_health", "work_life_balance", "writing", "yoga",
            "history", "politics", "animals"
        ],
        env="DEFAULT_DOMAINS"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[Path] = Field(default=None, env="LOG_FILE")
    verbose_logging: bool = Field(default=True, env="VERBOSE_LOGGING")  # Set to False in production
    
    # File Upload
    max_file_size: int = Field(default=150 * 1024 * 1024, env="MAX_FILE_SIZE")  # 150MB for large textbook PDFs
    allowed_file_types: List[str] = Field(
        default=[".txt", ".pdf", ".docx", ".md"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Image Extraction (Query-time)
    # Set to False to disable image extraction during queries (useful for fine-tuning)
    # Note: This only affects query-time extraction, not image extraction during document upload
    enable_image_extraction_during_query: bool = Field(
        default=True,
        env="ENABLE_IMAGE_EXTRACTION_DURING_QUERY"
    )
    
    # RAG Context Filtering (Query-time)
    # Set to False to use ALL retrieved documents for LLM (useful for fine-tuning)
    # When True: Only documents with 2+ keyword matches are used (current behavior)
    # When False: All retrieved documents are passed to LLM regardless of keyword matching
    filter_rag_context_by_relevance: bool = Field(
        default=False,
        env="FILTER_RAG_CONTEXT_BY_RELEVANCE"
    )
    
    # Hugging Face Token (for publishing datasets)
    hf_token: Optional[str] = Field(
        default=None,
        env="HF_TOKEN",
        description="Hugging Face token for publishing datasets to HF Hub"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_domain_path(domain: str) -> Path:
    """Get the vector store path for a specific domain."""
    return settings.vectorstore_path / domain


def get_emotion_model_path(model_name: str) -> Path:
    """Get the path for a specific emotion detection model."""
    return settings.emotion_models_path / model_name 