# Vectorstore Validation Guide

This guide explains how to validate and check the quality of your vectorstores.

## Quick Validation

### Using Python Script (Recommended)

**Validate a single domain:**
```bash
python scripts/validate_vectorstore.py general_health
```

**Validate all domains:**
```bash
python scripts/validate_vectorstore.py all
```

### Using the API

```bash
# Validate a domain
curl http://localhost:8000/api/vectorstore/validate/general_health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/vectorstore/validate/general_health" | ConvertTo-Json -Depth 10
```

## What Gets Validated

### 1. Basic Statistics ✅
- **Chunk count**: Number of document chunks in the vectorstore
- **Document count**: Number of unique documents
- **Database size**: Actual disk usage
- **Average chunks per document**: Data distribution

**What to check:**
- Chunk count should match expected number (based on your documents)
- Document count should match number of files uploaded
- Size should be reasonable (not bloated)

### 2. Data Integrity 🔍
- **Embeddings**: Presence and correct dimensions
- **Documents**: Text content available
- **Metadata**: File names, page numbers, hashes, etc.
- **Data completeness**: No missing fields

**What to check:**
- Embedding dimension should be 384 (for all-MiniLM-L6-v2)
- All chunks should have text content
- Metadata should include file_name, page, file_hash

### 3. Search Functionality 🔎
- **Query execution**: Can perform similarity searches
- **Results returned**: Queries return relevant chunks
- **Similarity scores**: Distance metrics are reasonable

**What to check:**
- Search queries should return results
- Similarity scores should be reasonable (lower = more similar for cosine distance)
- Results should be relevant to the query

### 4. Sample Query Quality 📝
- **Relevance**: Retrieved documents match the query
- **Score distribution**: Similarity scores are in expected range
- **Source tracking**: Can trace results back to original documents

**What to check:**
- Top results should be relevant
- Scores typically < 1.5 for good matches (cosine distance)
- Sources should be identifiable

## Validation Results

### Status Levels

- ✅ **Pass**: All checks passed, vectorstore is ready
- ⚠️ **Warning**: Some issues found, review recommended
- ❌ **Fail**: Critical issues, fix before use

### Example Output

```json
{
  "domain": "general_health",
  "timestamp": "2025-12-09T22:15:00",
  "checks": {
    "basic_stats": {
      "status": "pass",
      "chunk_count": 7318,
      "document_count": 9,
      "db_size": "85.47 MB"
    },
    "data_integrity": {
      "status": "pass",
      "has_embeddings": true,
      "has_documents": true,
      "embedding_dimension": 384,
      "expected_dimension": 384
    },
    "search_functionality": {
      "status": "pass",
      "test_query": "health symptoms",
      "results_returned": 3,
      "average_similarity_score": 0.8234
    }
  },
  "overall_status": "pass"
}
```

## Manual Quality Checks

### 1. Test Specific Queries

```python
from app.rag.domain_retrievers import get_domain_retriever

retriever = get_domain_retriever("general_health")

# Test query
results = retriever.similarity_search("diabetes symptoms", k=5)

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Source: {doc.metadata.get('file_name')}")
    print(f"  Page: {doc.metadata.get('page')}")
    print(f"  Content: {doc.page_content[:200]}...")
```

### 2. Check Similarity Scores

```python
# Get results with scores
results = retriever.vectorstore.similarity_search_with_score(
    "diabetes treatment", 
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc.metadata.get('file_name')}")
    # Lower scores = more similar (cosine distance)
    if score < 0.8:
        print("  ✅ Very relevant")
    elif score < 1.2:
        print("  ✅ Relevant")
    else:
        print("  ⚠️  Less relevant")
```

### 3. Verify Document Sources

```python
# Get all unique documents
collection = retriever.vectorstore._collection
all_data = collection.get(limit=10000)

unique_files = set()
for metadata in all_data.get("metadatas", []):
    if metadata and "file_name" in metadata:
        unique_files.add(metadata["file_name"])

print(f"Unique documents: {len(unique_files)}")
for file in sorted(unique_files):
    print(f"  - {file}")
```

## Common Issues and Fixes

### Issue: Low chunk count
**Symptom**: Fewer chunks than expected
**Fix**: 
- Check if documents were fully processed
- Verify chunk size settings (default: 500 chars)
- Re-upload documents if needed

### Issue: No search results
**Symptom**: Queries return empty results
**Fix**:
- Verify embeddings were created
- Check if vectorstore has data
- Try broader queries

### Issue: Poor relevance
**Symptom**: Results don't match queries
**Fix**:
- Adjust chunk size (smaller = more precise)
- Increase chunk overlap (more context)
- Check if source documents match domain

### Issue: Missing metadata
**Symptom**: Can't trace results to source files
**Fix**:
- Verify upload process includes metadata
- Check if file_name/file_hash are set
- Re-upload with proper metadata

## Best Practices

### Regular Validation
- **After bulk uploads**: Validate immediately
- **Before production**: Full validation check
- **Periodic checks**: Weekly/monthly reviews

### Performance Testing
- Test with domain-specific queries
- Verify response times (< 1 second)
- Check relevance scores distribution

### Data Quality
- Monitor chunk count trends
- Track average chunks per document
- Watch for size bloat (use VACUUM if needed)

## API Endpoints

### Validate Domain
```
GET /api/vectorstore/validate/{domain}
```

### Get Domain Stats
```
GET /api/vectorstore/domains
```

### Get Domain Info
```
GET /api/vectorstore/{domain}
```

## Automation

Add validation to your upload workflow:

```python
# After uploading documents
from app.rag.domain_retrievers import get_domain_retriever

retriever = get_domain_retriever("general_health")
stats = retriever.get_collection_stats()

# Quick validation
assert stats["chunk_count"] > 0, "No chunks created"
assert stats["document_count"] > 0, "No documents found"

# Test search
results = retriever.similarity_search("test query", k=1)
assert len(results) > 0, "Search not working"
```

## Troubleshooting

If validation fails:
1. Check logs for errors during upload
2. Verify document format (PDF, DOCX, etc.)
3. Check embedding model is loaded
4. Ensure vectorstore directory has write permissions
5. Try clearing and re-uploading

For detailed logs, check:
- `logs/meetara.log` - General application logs
- `logs/batch_upload_*.log` - Upload process logs

