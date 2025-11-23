# 🚀 Quick Start Guide

Get Meetara Core running in 5 minutes!

## Prerequisites

- **Python 3.12+** (or 3.10+)
- **8GB+ RAM**
- **5GB+ disk space**
- **Tesseract OCR** (optional, for image text extraction)

## Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd meetara-core
```

### Step 2: Setup Environment
```bash
# Create virtual environment
python -m venv .venv-meetara

# Activate (Windows)
.venv-meetara\Scripts\activate

# Activate (Linux/Mac)
source .venv-meetara/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (optional, for image text extraction)
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### Step 3: Configure
```bash
# Copy example env file
cp env.example .env

# ⚠️ IMPORTANT: Review these settings in .env
```

**🔧 Key Settings to Configure:**

1. **Model Selection** (required for best performance):
```env
USE_MEETARA_MODELS=true                           # Enable fine-tuned model
MEETARA_HF_MODEL_ID=meetara-lab/meetara-qwen3-1.7b-gguf  # Auto-downloads
```

2. **Performance Tuning** (optional):
```env
ENABLE_IMAGE_EXTRACTION_DURING_QUERY=false       # false = faster queries
FILTER_RAG_CONTEXT_BY_RELEVANCE=false            # false = more context
```

3. **For Development/Fine-tuning**:
```env
ENABLE_IMAGE_EXTRACTION_DURING_QUERY=true        # Include images in training
FILTER_RAG_CONTEXT_BY_RELEVANCE=false            # Full RAG context
```

💡 **Tip:** Use default settings for first run - they work out of the box!

### Step 4: Start Server
```bash
python main.py
```

**✅ Server running at `http://localhost:8000`**

## Quick Test

### Health Check
```bash
curl http://localhost:8000/health
```

### Chat Query
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I improve my sleep quality?",
    "session_id": "test-123",
    "context": {}
  }'
```

### Interactive API Docs
Open browser: http://localhost:8000/docs

## Model Configuration

### Option 1: Hugging Face (Recommended)
```env
MEETARA_HF_MODEL_ID=meetara-lab/meetara-qwen3-1.7b-gguf
MEETARA_HF_MODEL_FILE=meetara-qwen3-1.7b-Q4_K_M.gguf
```

Model downloads automatically on first request (~1.2 GB, cached).

### Option 2: Local Model
```env
MEETARA_MODELS_PATH=C:/path/to/models
MEETARA_INSTRUCT_MODEL=meetara-qwen3-1.7b-Q4_K_M.gguf
```

## Next Steps

1. **Upload Documents**: Use `/api/upload/doc` endpoint
2. **Test Domains**: Try queries across different domains
3. **Read Full Docs**: See `README.md` for complete documentation
4. **Explore API**: Visit http://localhost:8000/docs

## Troubleshooting

**Port in use?**
```bash
# Change port in .env
API_PORT=8001
```

**Model not loading?**
- Check internet connection (for HF download)
- Verify `.env` configuration
- Check logs: `logs/meetara.log`

**Need help?**
- Full documentation: `README.md`
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

**🎉 You're ready to go!** See `README.md` for complete documentation.
