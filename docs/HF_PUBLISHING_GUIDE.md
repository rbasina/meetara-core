# Publishing Vectorstores to Hugging Face Hub

This guide explains how to publish your Meetara Core vectorstores to Hugging Face Hub for sharing and distribution.

## Overview

Publishing vectorstores to Hugging Face Hub allows you to:
- Share pre-built knowledge bases with team members
- Distribute domain-specific embeddings easily
- Create backups in the cloud
- Enable users to download ready-to-use vectorstores

## Prerequisites

1. **Hugging Face Account**: Create one at https://huggingface.co/join
2. **Hugging Face Token**: Get your token from https://huggingface.co/settings/tokens
3. **Install Dependencies**:
   ```bash
   pip install huggingface-hub datasets
   ```
4. **Login to Hugging Face**:
   ```bash
   huggingface-cli login
   ```
   Or set environment variable:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

## Publishing Methods

### Method 1: Using PowerShell Script (Recommended)

**Easiest way to publish - handles token automatically:**

```powershell
# Basic usage (token from .env or HF login cache)
.\scripts\publish_vectorstore.ps1 -Domain "general_health" -RepoId "meetara-lab/vectorstore-general_health"

# With explicit token
.\scripts\publish_vectorstore.ps1 -Domain "general_health" -RepoId "meetara-lab/vectorstore-general_health" -Token "hf_..."

# Private repository
.\scripts\publish_vectorstore.ps1 -Domain "general_health" -RepoId "meetara-lab/vectorstore-general_health" -Private

# With custom commit message
.\scripts\publish_vectorstore.ps1 -Domain "general_health" -RepoId "meetara-lab/vectorstore-general_health" -CommitMessage "Updated with latest medical documents"
```

**Token Resolution (automatic, in order):**
1. `-Token` parameter (if provided)
2. `HF_TOKEN` from `.env` file
3. `HF_TOKEN` environment variable
4. Hugging Face login cache (`~/.huggingface/token` from `huggingface-cli login`)

### Method 2: Using the API

#### Publish a Single Domain

**PowerShell (Windows):**
```powershell
$body = @{
    domain = "general_health"
    repo_id = "meetara-lab/vectorstore-general_health"
    private = $false
    commit_message = "Initial publication of general health vectorstore"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/publish" `
    -Method POST `
    -Headers @{
        "Authorization" = "Bearer YOUR_HF_TOKEN"
        "Content-Type" = "application/json"
    } `
    -Body $body
```

**Bash/Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/api/vectorstore/publish" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "general_health",
    "repo_id": "meetara-lab/vectorstore-general_health",
    "private": false,
    "commit_message": "Initial publication of general health vectorstore"
  }'
```

#### Publish All Domains

**PowerShell (Windows):**
```powershell
$body = @{
    repo_prefix = "meetara-lab/vectorstore"
    private = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/publish-all" `
    -Method POST `
    -Headers @{
        "Authorization" = "Bearer YOUR_HF_TOKEN"
        "Content-Type" = "application/json"
    } `
    -Body $body
```

**Bash/Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/api/vectorstore/publish-all" \
  -H "Authorization: Bearer YOUR_HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_prefix": "meetara-lab/vectorstore",
    "private": false
  }'
```

#### Export Locally (Without Publishing)

**PowerShell (Windows):**
```powershell
$body = @{
    domain = "general_health"
    include_embeddings = $true
    include_documents = $true
    include_metadata = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/export" `
    -Method POST `
    -Headers @{"Content-Type" = "application/json"} `
    -Body $body
```

**Bash/Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/api/vectorstore/export" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "general_health",
    "include_embeddings": true,
    "include_documents": true,
    "include_metadata": true
  }'
```

### Method 3: Using Python Script

```python
from app.rag.hf_publisher import VectorstorePublisher

# Initialize publisher
publisher = VectorstorePublisher(token="your_hf_token")

# Publish a single domain
success, repo_url = publisher.publish_domain(
    domain="general_health",
    repo_id="meetara-lab/vectorstore-general_health",
    private=False
)

if success:
    print(f"✅ Published to: {repo_url}")
else:
    print("❌ Failed to publish")
```

### Method 4: Publish All Domains

```python
from app.rag.hf_publisher import VectorstorePublisher

publisher = VectorstorePublisher(token="your_hf_token")

results = publisher.publish_all_domains(
    repo_prefix="meetara-lab/vectorstore",
    private=False
)

for domain, (success, repo_url) in results.items():
    if success:
        print(f"✅ {domain}: {repo_url}")
    else:
        print(f"❌ {domain}: Failed")
```

## Loading Published Vectorstores

### Using Python Script

Load a vectorstore from Hugging Face Hub directly:

```python
from app.rag.hf_publisher import load_vectorstore_from_hub

# Load from Hub
vectorstore = load_vectorstore_from_hub(
    repo_id="meetara-lab/vectorstore-general_health",
    domain="general_health",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Use with LangChain
from langchain_chroma import Chroma
retriever = vectorstore.as_retriever()
results = retriever.invoke("your query here")
```

### From Hugging Face Hub (API)

```python
from app.rag.hf_publisher import load_vectorstore_from_hub

# Load from Hub
vectorstore = load_vectorstore_from_hub(
    repo_id="meetara-lab/vectorstore-general_health",
    domain="general_health",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Use with LangChain
from langchain_chroma import Chroma
retriever = vectorstore.as_retriever()
results = retriever.invoke("your query here")
```

### Using the API Endpoint

**PowerShell (Windows):**
```powershell
$body = @{
    repo_id = "meetara-lab/vectorstore-general_health"
    domain = "general_health"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/load-from-hub" `
    -Method POST `
    -Headers @{"Content-Type" = "application/json"} `
    -Body $body
```

**Bash/Linux/Mac:**
```bash
curl -X POST "http://localhost:8000/api/vectorstore/load-from-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "meetara-lab/vectorstore-general_health",
    "domain": "general_health"
  }'
```

### Using Hugging Face Datasets Directly

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("meetara-lab/vectorstore-general_health")

# Access data
for item in dataset["train"]:
    print(f"ID: {item['id']}")
    print(f"Document: {item['document'][:100]}...")
    print(f"Embedding shape: {item['embedding'].shape}")
```

## Best Practices

### 1. Repository Naming

Use consistent naming conventions:
- Format: `organization/vectorstore-domain_name`
- Example: `meetara-lab/vectorstore-general_health`
- Avoid spaces; use underscores or hyphens

### 2. Commit Messages

Write descriptive commit messages:
```json
{
  "commit_message": "Update general health vectorstore\n\n- Added 500 new medical documents\n- Updated embeddings with latest model\n- Total chunks: 15,234"
}
```

### 3. Version Control

- Each publish creates a new commit
- Use tags for major versions
- Include version info in README

### 4. Privacy

- Use `private: true` for sensitive data
- Review metadata before publishing
- Ensure no PII in documents

### 5. Dataset Cards

The system automatically generates dataset cards with:
- Domain information
- Statistics (chunk count, document count)
- Usage instructions
- Loading examples

## Dataset Format

The published dataset includes:

- **id**: Unique identifier for each chunk
- **embedding**: Vector embedding (numpy array, float32)
- **document**: Original text content
- **metadata**: JSON string with file_name, file_hash, page_number, etc.

## Troubleshooting

### "No Hugging Face token found"

```bash
# Option 1: Use CLI
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN="your_token_here"

# Option 3: Pass in API request
Authorization: Bearer your_token_here
```

### "Repository not found"

The repository will be created automatically. If it fails:
1. Check your token permissions
2. Verify the repository ID format
3. Check if repository already exists with different visibility

### "Dataset too large"

For very large datasets (>50GB):
1. Consider publishing domains separately
2. Use private repositories (no size limit)
3. Contact Hugging Face for large dataset support

### "Export failed"

Check:
1. Domain has documents (check with `/api/vectorstore/domains`)
2. Sufficient disk space for export
3. ChromaDB is accessible and not locked

## Example Workflow

1. **Export locally first** (test):

   **PowerShell:**
   ```powershell
   $body = @{domain = "general_health"} | ConvertTo-Json
   Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/export" `
       -Method POST -Headers @{"Content-Type" = "application/json"} -Body $body
   ```

   **Bash:**
   ```bash
   curl -X POST "http://localhost:8000/api/vectorstore/export" \
     -d '{"domain": "general_health"}'
   ```

2. **Verify export** (check files)

3. **Publish to Hub**:

   **PowerShell:**
   ```powershell
   $token = $env:HF_TOKEN  # or use: "your_token_here"
   $body = @{
       domain = "general_health"
       repo_id = "meetara-lab/vectorstore-general_health"
   } | ConvertTo-Json
   
   Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/publish" `
       -Method POST `
       -Headers @{
           "Authorization" = "Bearer $token"
           "Content-Type" = "application/json"
       } `
       -Body $body
   ```

   **Bash:**
   ```bash
   curl -X POST "http://localhost:8000/api/vectorstore/publish" \
     -H "Authorization: Bearer $HF_TOKEN" \
     -d '{
       "domain": "general_health",
       "repo_id": "meetara-lab/vectorstore-general_health"
     }'
   ```

4. **Share with team**:
   - Share repository URL
   - Users can load directly from Hub

## API Reference

### POST `/api/vectorstore/publish`

Publish a domain to Hugging Face Hub.

**Request:**
```json
{
  "domain": "general_health",
  "repo_id": "meetara-lab/vectorstore-general_health",
  "private": false,
  "create_pr": false,
  "commit_message": "Optional custom message"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully published...",
  "repo_url": "https://huggingface.co/datasets/...",
  "domain": "general_health",
  "stats": {...}
}
```

### POST `/api/vectorstore/export`

Export locally without publishing.

### POST `/api/vectorstore/load-from-hub`

Load a published vectorstore from Hub.

### GET `/api/vectorstore/domains`

Get list of all available domains.

## Additional Resources

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [Datasets Library Documentation](https://huggingface.co/docs/datasets)
- [Hugging Face Hub Python API](https://huggingface.co/docs/huggingface_hub)

