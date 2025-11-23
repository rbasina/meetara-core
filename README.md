# 🚀 Meetara Core

**Production-ready RAG (Retrieval-Augmented Generation) backend** with emotion-aware responses, multi-domain support, and Hugging Face model integration.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Model Management](#-model-management)
- [Development](#-development)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

**Meetara Core** is a modular, production-ready RAG assistant backend that provides:

- ✅ **100+ Domain Support** - Healthcare, Education, Business, Technology, and more
- ✅ **Emotion-Aware Responses** - Facial and speech emotion detection
- ✅ **Multi-Language Support** - 15+ languages including Telugu, Tamil, Bengali
- ✅ **Offline-First** - All processing done locally with open-source models
- ✅ **Hugging Face Integration** - Automatic model download and caching
- ✅ **Production Ready** - FastAPI, comprehensive error handling, monitoring

---

## ✨ Features

### Core Capabilities

- **Multi-Domain RAG**: Handle 100+ domains using vector search with ChromaDB
- **LangChain Agent System**: Tool-based orchestration with intelligent routing
- **Emotion-Aware Responses**: Facial and speech emotion detection for adaptive responses
- **Multi-Language Support**: Telugu, Tamil, Bengali + 12 other languages
- **Smart Domain Detection**: Context-aware keyword matching with cross-domain intelligence
- **Image Extraction**: Automatic extraction and retrieval of images from documents
- **Conversation Context**: Maintains domain context across multi-turn conversations

### Technical Features

- **Vector Storage**: ChromaDB with Snappy compression (75-85% storage reduction)
- **Model Caching**: Single-instance model loading with Hugging Face cache integration
- **Performance**: Shared embeddings model, batch APIs, parallel loading
- **Config-Driven**: All domain logic via YAML files (no hardcoding)
- **Modular Architecture**: Extensible design for easy customization

---

## 🚀 Quick Start

> 📖 **For detailed quick start guide, see [docs/QUICK_START.md](docs/QUICK_START.md)**

### Prerequisites

- **Python 3.12+** (recommended) or Python 3.10+
- **8GB+ RAM** (for model loading)
- **5GB+ disk space**
- **Git** (for cloning)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd meetara-core
```

2. **Create virtual environment**
```bash
python -m venv .venv-meetara
.venv-meetara\Scripts\activate  # Windows
# source .venv-meetara/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Tesseract OCR (for image text extraction)**

**Windows:**
```bash
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
# Or use chocolatey:
choco install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

> 📝 **Note**: Tesseract is required for extracting text from images in PDFs. The system will work without it, but OCR will be skipped.

5. **Configure environment**
```bash
# Copy example env file
cp env.example .env

# Edit .env and set your configuration (see Configuration section)
```

6. **Start the server**
```bash
python main.py
```

7. **Verify installation**
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy", ...}
```

**🎉 API available at `http://localhost:8000`**

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🏗️ Architecture

> 📖 **For complete architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Request                          │
└────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Endpoints (/api/chat)              │
└────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MeetaraAgent (LangChain Agent)             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Query        │  │ Domain       │  │ RAG          │ │
│  │ Analyzer     │→ │ Detection    │→ │ Retrieval    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         GGUF LLM Processor (meetara-qwen3-1.7b)         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ • Hugging Face Model Download & Caching          │   │
│  │ • Single-Instance Loading (reused for all reqs) │   │
│  │ • Domain-Specific Prompt Generation             │   │
│  │ • Response Structure Validation                 │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Structured Response + Images               │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
meetara-core/
├── app/
│   ├── api/                    # FastAPI route handlers
│   │   ├── chat.py            # Chat endpoint with RAG
│   │   ├── emotion.py         # Emotion detection APIs
│   │   ├── upload.py          # Document upload & management
│   │   └── image_generation.py # Image generation API
│   ├── agent/                  # LangChain agent system
│   │   ├── planner.py         # Main agent orchestration
│   │   ├── mcp_router.py      # Multi-component planner
│   │   └── tools/             # Agent tools
│   │       ├── adapter_selector.py    # Domain routing
│   │       ├── translation_tool.py    # Multi-language
│   │       ├── speech_tool.py          # STT/TTS
│   │       ├── emotion_tool.py         # Speech emotion
│   │       └── face_emotion_tool.py   # Facial emotion
│   ├── rag/                   # RAG components
│   │   ├── domain_retrievers.py      # Domain-specific retrievers
│   │   ├── semantic_domain_detector.py # Semantic domain detection
│   │   └── vector_loader.py          # Vector store management
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   ├── config_loader.py   # YAML config loader
│   │   ├── gguf_llm_processor.py # GGUF model processor
│   │   ├── llm_processor.py    # LLM integration
│   │   ├── query_analyzer.py  # Query analysis & domain detection
│   │   ├── domain_categorizer.py # Domain categorization
│   │   ├── logger.py          # Logging system
│   │   └── security.py        # Security & validation
│   ├── services/              # Service layer
│   │   └── image_generator.py # Image generation service
│   └── utils/                 # Utility functions
├── config/                    # Configuration files
│   ├── domain_config.yaml     # Domain definitions & settings
│   ├── domain_keywords.yaml   # Domain keywords for detection
│   └── tier_config.yaml       # Domain tier configurations
├── scripts/                   # Utility scripts
│   ├── batch_uploader.py      # Batch document upload
│   ├── check_db_size.py       # Database size checker
│   └── setup_environment.py   # Environment setup
├── tests/                     # Test files
├── docs/                      # Documentation
│   └── archive/               # Archived documentation
├── vectorstore/               # ChromaDB vector stores (per domain)
├── images/                    # Extracted document images
├── models/                    # Emotion detection models
├── main.py                    # FastAPI application entry point
├── requirements.txt           # Python dependencies
├── env.example                # Environment variables template
└── README.md                  # This file
```

### Data Flow

1. **User Query** → FastAPI endpoint (`/api/chat/`)
2. **Query Analysis** → Domain detection (keyword + semantic)
3. **RAG Retrieval** → Vector search in domain-specific ChromaDB
4. **Context Building** → Merge RAG context + conversation history
5. **LLM Generation** → GGUF model generates structured response
6. **Response Formatting** → Add images, format markdown, return JSON

---

## ⚙️ Configuration

### 🔧 Critical Configuration Flags

These are the **most important settings** you should understand and configure:

#### 1. **Model Configuration** ⚡

```env
# Enable/disable Meetara fine-tuned models (recommended: true)
USE_MEETARA_MODELS=true

# Hugging Face model (auto-downloads and caches)
MEETARA_HF_MODEL_ID=meetara-lab/meetara-qwen3-1.7b-gguf
MEETARA_HF_MODEL_FILE=meetara-qwen3-1.7b-Q4_K_M.gguf
```

**What it does:**
- ✅ `true` → Uses your fine-tuned Meetara model (better responses, domain-aware)
- ❌ `false` → Falls back to base models (less accurate)
- 💡 **Recommendation:** Keep `true` for production

#### 2. **Image Extraction Control** 🖼️

```env
# Extract images during query time (affects performance)
ENABLE_IMAGE_EXTRACTION_DURING_QUERY=false
```

**What it does:**
- ✅ `true` → Extracts relevant images during each query (slower, visual context)
- ❌ `false` → Only extracts during document upload (faster queries)
- 💡 **Recommendation:** 
  - `false` for production (better performance)
  - `true` for fine-tuning data generation (better training data)

#### 3. **RAG Context Filtering** 🎯

```env
# Filter RAG context by keyword relevance
FILTER_RAG_CONTEXT_BY_RELEVANCE=false
```

**What it does:**
- ✅ `true` → Filters retrieved documents by keyword matching (stricter)
- ❌ `false` → Returns all semantically similar documents (more context)
- 💡 **Recommendation:**
  - `false` for training data generation (full context)
  - `true` for production (focused responses)

#### 4. **Domain Response Structure** 📋

**Configured in:** `app/core/domain_categorizer.py` → `DOMAIN_SECTIONS`

**What it does:**
- Defines response structure per domain category
- Healthcare → diagnosis, symptoms, treatment, precautions
- Education → explanation, examples, practice_questions
- Legal → legal_context, rights, obligations, next_steps

**Note:** This is **NOT** for model selection (one model for all domains). It only controls response formatting structure.

---

### Environment Variables (.env)

Create a `.env` file from `env.example`:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://localhost:2025"]

# Vector Store Configuration
VECTORSTORE_PATH=vectorstore
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=100

# Meetara GGUF Models Configuration
# Option 1: Use Hugging Face model (recommended - automatic download)
MEETARA_HF_MODEL_ID=meetara-lab/meetara-qwen3-1.7b-gguf
MEETARA_HF_MODEL_FILE=meetara-qwen3-1.7b-Q4_K_M.gguf
# Cache directory (leave empty to use default HF cache: ~/.cache/huggingface/hub)
# On Windows: C:\Users\<username>\.cache\huggingface\hub
# MEETARA_MODEL_CACHE_DIR=

# Option 2: Use local models (fallback if HF not set)
# MEETARA_MODELS_PATH=C:/path/to/models
# MEETARA_INSTRUCT_MODEL=meetara-qwen3-1.7b-Q4_K_M.gguf

USE_MEETARA_MODELS=true
LOCAL_LLM_MAX_LENGTH=640
LOCAL_LLM_TOP_P=0.9
LOCAL_LLM_TOP_K=50

# LLM Configuration
LLM_CONTEXT_LENGTH=4096
LLM_TEMPERATURE=0.7

# Speech Processing
STT_MODEL=base
TTS_VOICE=en-US-JennyNeural
AUDIO_SAMPLE_RATE=16000

# Emotion Detection
EMOTION_MODELS_PATH=models/emotion
FACE_DETECTION_CONFIDENCE=0.5

# Fine-tuning Configuration (for meetara-lab integration)
ENABLE_IMAGE_EXTRACTION_DURING_QUERY=false
FILTER_RAG_CONTEXT_BY_RELEVANCE=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/meetara.log
```

### Domain Configuration

Domains are configured in `config/domain_config.yaml`:

```yaml
categories:
  healthcare:
    tier: safety_critical
    validation_level: strict
    domains:
      general_health:
        requires_validation: true
        priority: 1
      mental_health:
        requires_validation: true
        priority: 1
```

**Adding a new domain:**
1. Add domain to `config/domain_config.yaml`
2. Add keywords to `config/domain_keywords.yaml`
3. Upload documents via `/api/upload/doc`
4. Domain automatically available for queries

---

## 📚 API Documentation

### Core Endpoints

#### Chat Query
```bash
POST /api/chat/
Content-Type: application/json

{
  "query": "How can I improve my sleep quality?",
  "session_id": "conv-abc123",  # Reuse same ID for conversation
  "context": {
    "domain": "general_health",  # Optional: specify domain
    "lang": "en",
    "emotion": "anxious"
  }
}
```

**Response:**
```json
{
  "response": "**Quick Answer:** To improve sleep quality...",
  "domain": "general_health",
  "confidence": 0.95,
  "images": [
    {
      "image_url": "/api/images/general_health/sleep_guide.png",
      "caption": "Sleep hygiene diagram"
    }
  ],
  "sources": ["Sleep Guide.pdf", "Health Manual.pdf"]
}
```

#### Get Available Domains
```bash
GET /api/chat/domains/categorized
```

#### Document Upload

**Command Line (cURL):**
```bash
# Upload with domain specified
curl -X POST "http://localhost:8000/api/upload/doc" \
  -F "file=@document.pdf" \
  -F "domain=general_health"

# Auto-detect domain
curl -X POST "http://localhost:8000/api/upload/doc" \
  -F "file=@document.pdf"
```

**Python Script:**
```bash
# Single file upload
python scripts/batch_uploader.py --file document.pdf --domain general_health

# Batch upload from directory
python scripts/batch_uploader.py --source-dir downloads/ --domain general_health

# Auto-detect domain
python scripts/batch_uploader.py --file document.pdf --auto-detect
```

**UI (Swagger):**
1. Open http://localhost:8000/docs
2. Navigate to `/api/upload/doc`
3. Click "Try it out"
4. Upload file and execute

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#-document-upload-guide) for complete upload guide.

#### Health Check
```bash
GET /health
```

### Complete API Reference

Visit http://localhost:8000/docs for interactive Swagger documentation.

---

## 🤖 Model Management

### Hugging Face Model Integration

Meetara Core supports automatic model download from Hugging Face:

**Configuration:**
```env
MEETARA_HF_MODEL_ID=meetara-lab/meetara-qwen3-1.7b-gguf
MEETARA_HF_MODEL_FILE=meetara-qwen3-1.7b-Q4_K_M.gguf
```

**How it works:**
1. **First Request**: Downloads model from HF (~1.2 GB, one-time)
2. **Caching**: Stores in `~/.cache/huggingface/hub` (standard HF cache)
3. **Loading**: Loads model once into memory
4. **Reuse**: Same model instance used for all requests

**Performance:**
- First request: ~2-3 minutes (download + load)
- Subsequent requests: ~30-35 seconds (generation only)

### Model Loading Flow

```
Request → Check Cache → Download (if needed) → Load (once) → Reuse
```

### Custom Cache Location

To use a custom cache directory:
```env
# Example: Use a custom directory for model cache
MEETARA_MODEL_CACHE_DIR=/path/to/custom/cache

# Default: Uses standard HF cache location
# Windows: C:\Users\<username>\.cache\huggingface\hub
# Linux/Mac: ~/.cache/huggingface/hub
```

---

## 🔧 Development

### Adding a New Domain

1. **Add to `config/domain_config.yaml`:**
```yaml
categories:
  your_category:
    tier: quality
    domains:
      your_domain:
        requires_validation: false
        priority: 2
```

2. **Add keywords to `config/domain_keywords.yaml`:**
```yaml
your_domain:
  keywords: ["keyword1", "keyword2", "phrase"]
```

3. **Upload documents:**
```bash
curl -X POST http://localhost:8000/api/upload/doc \
  -F "file=@your_document.pdf" \
  -F "domain=your_domain"
```

**No code changes required!** Everything is config-driven.

### Adding a New Tool

1. Create tool in `app/agent/tools/your_tool.py`:
```python
from langchain_core.tools import BaseTool

class YourTool(BaseTool):
    name = "your_tool"
    description = "Tool description"
    
    def _run(self, query: str) -> str:
        # Tool logic
        return result
```

2. Add to agent in `app/agent/planner.py`:
```python
tools.append(YourTool())
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_chat_api.py
```

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t meetara-core .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/vectorstore:/app/vectorstore \
  -v $(pwd)/.env:/app/.env \
  meetara-core
```

### Docker Compose

```bash
docker-compose up -d
```

### Production Considerations

- **Environment Variables**: Set all sensitive configs via `.env`
- **Model Caching**: Ensure `~/.cache/huggingface/hub` is persistent
- **Vector Store**: Mount `vectorstore/` directory as volume
- **Logging**: Configure log rotation and monitoring
- **Health Checks**: Use `/health` endpoint for monitoring

---

## 🔍 Troubleshooting

### Model Not Loading

**Issue**: Model download fails or model not found

**Solutions:**
1. Check internet connection (for HF download)
2. Verify `MEETARA_HF_MODEL_ID` and `MEETARA_HF_MODEL_FILE` in `.env`
3. Check disk space (~2 GB needed)
4. Verify cache directory permissions
5. Check logs: `logs/meetara.log`

### Domain Not Detected

**Issue**: Query routed to wrong domain

**Solutions:**
1. Check `config/domain_keywords.yaml` has relevant keywords
2. Verify domain exists in `config/domain_config.yaml`
3. Use `context.domain` to explicitly specify domain
4. Check logs for domain detection scores

### Slow Response Times

**Issue**: Responses take too long

**Solutions:**
1. First request includes model download (one-time)
2. Reduce `LOCAL_LLM_MAX_LENGTH` in `.env`
3. Reduce `LLM_CONTEXT_LENGTH` in `.env`
4. Check CPU/RAM usage
5. Consider GPU acceleration if available

### Port Already in Use

**Issue**: Port 8000 already in use

**Solutions:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or change port in .env
API_PORT=8001
```

---

## 📊 Performance Metrics

### Typical Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Model Download | ~99s | One-time only |
| Model Load | ~1.6s | One-time only |
| Domain Detection | ~0.2-0.4s | Per request |
| RAG Retrieval | ~0.2-0.5s | Per request |
| LLM Generation | ~30-45s | Per request |
| **Total (cached)** | **~30-35s** | After first load |

### Optimization Tips

- **Model Caching**: Model loads once, reused for all requests
- **Vector Store**: ChromaDB with Snappy compression (75-85% reduction)
- **Shared Embeddings**: Single embeddings model instance
- **Batch Operations**: Use `/api/vectorstore/all` for bulk stats

---

## 🔒 Security

### Security Features

- **Input Sanitization**: XSS and injection protection
- **PII Redaction**: Automatic privacy protection in logs
- **File Validation**: Secure file upload handling
- **CORS Configuration**: Cross-origin request control
- **Rate Limiting**: Built-in request throttling

### Best Practices

- Never commit `.env` file to git
- Use environment variables for sensitive data
- Regularly update dependencies
- Monitor logs for suspicious activity
- Use HTTPS in production

---

## 📖 Additional Resources

### Configuration Files

- `config/domain_config.yaml` - Domain definitions and settings
- `config/domain_keywords.yaml` - Domain keywords for detection
- `config/tier_config.yaml` - Domain tier configurations
- `env.example` - Environment variables template

### Scripts

- `scripts/batch_uploader.py` - Batch document upload utility
- `scripts/check_db_size.py` - Database size checker
- `scripts/setup_environment.py` - Automated environment setup

### Documentation

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Archived Docs**: `docs/archive/` (historical documentation)

---

## 🤝 Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed contribution guidelines.

Quick steps:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

### Code Style

- **Formatting**: Black
- **Imports**: isort
- **Type Hints**: Required for all functions
- **Docstrings**: Required for all functions/classes

---

## 📄 License

MIT License - see LICENSE file for details

---

## 📚 Documentation

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Complete system architecture with flow diagrams
- **[Contributing Guide](docs/CONTRIBUTING.md)** - How to contribute to the project
- **[Data Management Guide](docs/DATA_MANAGEMENT.md)** - Managing downloads, images, and data directories
- **[Git Setup Guide](docs/GIT_SETUP.md)** - Setting up Git LFS for compressed data files
- **API Docs**: http://localhost:8000/docs (Swagger UI)

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/meetara-core/issues)
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 🎉 Status

**✅ Production Ready**

- All endpoints tested and working
- Model download and caching operational
- Multi-domain RAG functioning
- Emotion detection integrated
- Multi-language support active
- Documentation complete

---

**Meetara Core** - Empowering AI assistants with emotion-aware intelligence.

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Python**: 3.12+
