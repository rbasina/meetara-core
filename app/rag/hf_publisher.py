"""
Hugging Face Hub Publisher for Vector Databases.

This module provides functionality to export ChromaDB vectorstores to
Hugging Face Dataset format and publish them to the Hugging Face Hub.
Follows best practices for vector database sharing.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

try:
    from huggingface_hub import (
        HfApi,
        create_repo,
        upload_file,
        HfFolder,
        list_repo_files
    )
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from datasets import Dataset, DatasetDict, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from app.core.config import settings
from app.core.logger import rag_logger
from app.rag.domain_retrievers import get_domain_retriever, list_available_domains


class VectorstorePublisher:
    """Publisher for exporting and uploading vectorstores to Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize the publisher.
        
        Args:
            token: Hugging Face token. If None, uses token from HfFolder or environment.
                  Will automatically use cached token from 'huggingface-cli login' if available.
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for publishing. "
                "Install with: pip install huggingface-hub"
            )
        
        # Get token from parameter, environment, or HF cache
        self.token = token or HfFolder.get_token()
        
        if not self.token:
            rag_logger.warning(
                "No Hugging Face token found. "
                "Will attempt to use cached token from 'huggingface-cli login'. "
                "If that fails, set HF_TOKEN environment variable or pass token parameter."
            )
        
        # Initialize API - it will use cached token if self.token is None
        self.api = HfApi(token=self.token)
    
    def export_domain_to_dataset(
        self,
        domain: str,
        output_dir: Optional[Path] = None,
        include_embeddings: bool = True,
        include_documents: bool = True,
        include_metadata: bool = True,
        batch_size: int = 10000
    ) -> Optional[Dataset]:
        """Export a domain's vectorstore to Hugging Face Dataset format.
        
        Args:
            domain: Domain name to export
            output_dir: Directory to save dataset files. If None, uses temp directory.
            include_embeddings: Whether to include embedding vectors
            include_documents: Whether to include document text
            include_metadata: Whether to include metadata
            batch_size: Number of records to process per batch
            
        Returns:
            Dataset object or None if export fails
        """
        try:
            rag_logger.info(f"Starting export of domain '{domain}' to HF Dataset format...")
            
            retriever = get_domain_retriever(domain)
            if not retriever.vectorstore:
                rag_logger.error(f"Vectorstore not initialized for domain: {domain}")
                return None
            
            collection = retriever.vectorstore._collection
            total_count = collection.count()
            
            if total_count == 0:
                rag_logger.warning(f"Domain '{domain}' has no documents to export")
                return None
            
            rag_logger.info(f"Exporting {total_count:,} chunks from domain '{domain}'...")
            
            # Collect all data in batches
            all_ids = []
            all_embeddings = []
            all_documents = []
            all_metadatas = []
            
            # Process in batches to handle large vectorstores
            offset = 0
            batch_num = 0
            
            while offset < total_count:
                batch_num += 1
                batch_limit = min(batch_size, total_count - offset)
                
                rag_logger.info(
                    f"Processing batch {batch_num}: "
                    f"chunks {offset + 1:,}-{offset + batch_limit:,} of {total_count:,}"
                )
                
                # Get batch with all data
                include_list = []
                if include_embeddings:
                    include_list.append("embeddings")
                if include_documents:
                    include_list.append("documents")
                if include_metadata:
                    include_list.append("metadatas")
                
                results = collection.get(
                    limit=batch_limit,
                    offset=offset,
                    include=include_list if include_list else ["metadatas"]
                )
                
                batch_ids = results.get("ids", [])
                if not batch_ids:
                    break
                
                all_ids.extend(batch_ids)
                
                if include_embeddings and "embeddings" in results:
                    all_embeddings.extend(results["embeddings"])
                elif include_embeddings:
                    # If embeddings not included in results, try to get them separately
                    embeddings_results = collection.get(
                        ids=batch_ids,
                        include=["embeddings"]
                    )
                    all_embeddings.extend(embeddings_results.get("embeddings", []))
                
                if include_documents and "documents" in results:
                    all_documents.extend(results["documents"])
                elif include_documents:
                    docs_results = collection.get(
                        ids=batch_ids,
                        include=["documents"]
                    )
                    all_documents.extend(docs_results.get("documents", []))
                
                if include_metadata and "metadatas" in results:
                    # Convert metadata dicts to JSON strings for better compatibility
                    # Filter out very large fields to reduce dataset size
                    metadatas = results["metadatas"]
                    filtered_metadatas = []
                    for m in metadatas:
                        if not m:
                            filtered_metadatas.append("{}")
                            continue
                        
                        # Create a cleaned copy, removing very large fields
                        cleaned_meta = {}
                        large_fields = ['associated_images', 'full_text', 'raw_content', 'page_image']
                        for key, value in m.items():
                            # Skip fields that are known to be very large
                            if key in large_fields:
                                continue
                            # Skip any field that's a string longer than 10KB when serialized
                            if isinstance(value, str) and len(value) > 10000:
                                continue
                            # Skip nested dicts/lists that serialize to more than 10KB
                            try:
                                serialized = json.dumps(value)
                                if len(serialized) > 10000:
                                    continue
                            except (TypeError, ValueError):
                                pass  # Skip non-serializable values
                            cleaned_meta[key] = value
                        
                        filtered_metadatas.append(json.dumps(cleaned_meta))
                    all_metadatas.extend(filtered_metadatas)
                
                offset += batch_limit
                
                # Log progress
                progress_pct = (offset / total_count) * 100
                rag_logger.info(f"Progress: {progress_pct:.1f}% ({offset:,}/{total_count:,} chunks)")
            
            # Prepare dataset dictionary
            dataset_dict = {
                "id": all_ids,
            }
            
            if include_embeddings and all_embeddings:
                # Convert embeddings to lists (not numpy arrays) for better Parquet compatibility
                # Numpy arrays in dicts can cause serialization issues
                dataset_dict["embedding"] = [
                    emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
                    for emb in all_embeddings
                ]
                rag_logger.info(f"  ✓ Included {len(all_embeddings):,} embeddings")
            
            if include_documents and all_documents:
                dataset_dict["document"] = all_documents
                rag_logger.info(f"  ✓ Included {len(all_documents):,} documents")
            
            if include_metadata and all_metadatas:
                dataset_dict["metadata"] = all_metadatas
                rag_logger.info(f"  ✓ Included {len(all_metadatas):,} metadata entries")
            
            # Create dataset
            if DATASETS_AVAILABLE:
                # Create dataset with explicit features for Parquet compatibility
                from datasets import Features, Value, Sequence
                
                try:
                    features = Features({"id": Value("string")})
                    if include_embeddings and all_embeddings:
                        emb_dim = len(all_embeddings[0]) if all_embeddings else 384
                        features["embedding"] = Sequence(Value("float32"), length=emb_dim)
                    if include_documents and all_documents:
                        features["document"] = Value("string")
                    if include_metadata and all_metadatas:
                        features["metadata"] = Value("string")
                    
                    # Create dataset once
                    dataset = Dataset.from_dict(dataset_dict, features=features)
                    
                    if dataset is None:
                        raise ValueError("Dataset.from_dict returned None")
                    
                    # Clear format attributes to prevent Parquet errors
                    for attr in ['_format_columns', '_format_kwargs', '_format_type', '_output_all_columns']:
                        if hasattr(dataset, attr):
                            setattr(dataset, attr, None if attr != '_format_kwargs' else {})
                    
                    # Set metadata (optional)
                    try:
                        if not hasattr(dataset, 'info') or dataset.info is None:
                            from datasets import DatasetInfo
                            dataset.info = DatasetInfo()
                        dataset.info.description = f"Vector embeddings for domain: {domain}"
                        dataset.info.dataset_name = f"meetara-vectorstore-{domain}"
                    except Exception as meta_err:
                        rag_logger.debug(f"Could not set metadata (optional): {meta_err}")
                    
                    # Save to disk if requested
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        dataset.save_to_disk(str(output_dir))
                        rag_logger.info(f"Dataset saved to: {output_dir}")
                    
                    rag_logger.info(
                        f"✅ Successfully exported domain '{domain}': "
                        f"{total_count:,} chunks, "
                        f"{len(all_embeddings) if all_embeddings else 0:,} embeddings"
                    )
                    
                    return dataset
                except Exception as ds_error:
                    rag_logger.error(f"Error creating dataset: {ds_error}", exc_info=True)
                    raise ValueError(f"Failed to create dataset: {ds_error}") from ds_error
            else:
                # Fallback: Save as JSON/Parquet without datasets library
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save as JSON (simpler, but less efficient)
                    output_file = output_dir / f"{domain}_vectorstore.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "ids": all_ids,
                            "embeddings": [
                                emb.tolist() if isinstance(emb, np.ndarray) else emb
                                for emb in all_embeddings
                            ] if all_embeddings else [],
                            "documents": all_documents,
                            "metadatas": [json.loads(m) if isinstance(m, str) else m for m in all_metadatas],
                            "metadata": {
                                "domain": domain,
                                "embedding_model": settings.embedding_model,
                                "chunk_size": settings.chunk_size,
                                "chunk_overlap": settings.chunk_overlap,
                                "export_date": datetime.now().isoformat(),
                                "total_chunks": total_count
                            }
                        }, f, indent=2)
                    
                    rag_logger.info(f"Exported to JSON: {output_file}")
                
                return None
                
        except Exception as e:
            rag_logger.error(f"Failed to export domain '{domain}': {e}", exc_info=True)
            return None
    
    def _get_domain_examples(self, domain: str) -> Dict[str, List[str]]:
        """Get domain-specific example queries and use cases."""
        examples_map = {
            "general_health": {
                "example_queries": [
                    "What are the symptoms of diabetes?",
                    "How to manage hypertension?",
                    "What is the treatment for arthritis?",
                    "Explain cardiovascular health",
                    "How to read blood test results?"
                ],
                "use_cases": [
                    "Medical information retrieval",
                    "Health symptom checking",
                    "Medication information lookup",
                    "Disease education and awareness",
                    "Preventive care guidance"
                ]
            },
            "mental_health": {
                "example_queries": [
                    "What are the early warning signs of depression and when should I seek help?",
                    "What are effective cognitive behavioral techniques for managing panic attacks?",
                    "How can I support a loved one who is experiencing PTSD symptoms?",
                    "What are evidence-based strategies for managing chronic anxiety in daily life?",
                    "How does trauma therapy work and what can I expect from treatment?"
                ],
                "use_cases": [
                    "Mental wellness assessment and guidance",
                    "Evidence-based therapy information and techniques",
                    "Self-care and coping strategy development",
                    "Crisis intervention and support resources",
                    "Psychological education and awareness"
                ]
            },
            "women_health": {
                "example_queries": [
                    "What are the recommended screening guidelines for breast cancer?",
                    "How do hormonal changes during menopause affect bone density?",
                    "What are the signs and symptoms of polycystic ovary syndrome (PCOS)?",
                    "How can I manage endometriosis pain and symptoms?",
                    "What are the pregnancy care recommendations for each trimester?"
                ],
                "use_cases": [
                    "Women's health screening and prevention",
                    "Reproductive health information",
                    "Pregnancy and postpartum care guidance",
                    "Hormonal health management",
                    "Gynecological condition education"
                ]
            },
            "business": {
                "example_queries": [
                    "How to start a new business?",
                    "Marketing strategy for startups",
                    "Sales techniques for B2B",
                    "Project management best practices",
                    "Financial planning for entrepreneurs"
                ],
                "use_cases": [
                    "Business planning",
                    "Entrepreneurship guidance",
                    "Marketing strategies",
                    "Operations optimization",
                    "Leadership development"
                ]
            },
            "technology": {
                "example_queries": [
                    "Python programming best practices",
                    "Web development with React",
                    "Machine learning algorithms",
                    "Database design principles",
                    "Cybersecurity best practices"
                ],
                "use_cases": [
                    "Software development",
                    "Technical documentation lookup",
                    "Programming tutorials",
                    "System architecture",
                    "DevOps practices"
                ]
            },
            "nutrition": {
                "example_queries": [
                    "What are healthy meal plans?",
                    "How to read nutrition labels?",
                    "Dietary requirements for diabetes",
                    "Weight management strategies",
                    "Vitamin and mineral information"
                ],
                "use_cases": [
                    "Meal planning",
                    "Dietary guidance",
                    "Nutritional education",
                    "Health-conscious eating",
                    "Special diet information"
                ]
            },
            "academic_tutoring": {
                "example_queries": [
                    "How to solve quadratic equations?",
                    "Explain photosynthesis process",
                    "What is the structure of an essay?",
                    "How to study effectively for exams?",
                    "Explain the causes of World War I"
                ],
                "use_cases": [
                    "Homework help and explanations",
                    "Study guide creation",
                    "Concept clarification",
                    "Exam preparation",
                    "Subject-specific tutoring"
                ]
            }
        }
        
        # Try exact match first
        if domain in examples_map:
            return examples_map[domain]
        
        # Try partial match (e.g., "mental_health" matches "mental_health")
        for key, value in examples_map.items():
            if key in domain or domain in key:
                return value
        
        # Default examples for unknown domains
        return {
            "example_queries": [
                f"What information is available about {domain.replace('_', ' ')}?",
                f"How to get started with {domain.replace('_', ' ')}?",
                f"Best practices for {domain.replace('_', ' ')}",
                f"Common questions about {domain.replace('_', ' ')}",
                f"Resources for {domain.replace('_', ' ')}"
            ],
            "use_cases": [
                "Information retrieval",
                "Educational content lookup",
                "Best practices reference",
                "Resource discovery",
                "Domain-specific queries"
            ]
        }
    
    def _get_related_datasets_section(
        self,
        current_domain: str,
        repo_prefix: str = "meetara-lab/vectorstore"
    ) -> str:
        """Dynamically generate related datasets section based on available domains.
        
        Args:
            current_domain: The domain being published (to exclude from list)
            repo_prefix: Repository prefix for dataset links
            
        Returns:
            Markdown string with related datasets section
        """
        try:
            # Get all available domains locally (function already imported at top)
            all_domains = list_available_domains()
            
            # Remove current domain from list
            other_domains = [d for d in all_domains if d != current_domain]
            
            if not other_domains:
                return (
                    "## Alternatives and Related Datasets\n\n"
                    "Looking for other domains? More meeTARA vectorstore datasets are coming soon!\n\n"
                    "**🔗 View all meeTARA datasets**: "
                    "[https://huggingface.co/meetara-lab](https://huggingface.co/meetara-lab)\n"
                )
            
            # Try to get stats for each domain (optional - won't fail if domain not loaded)
            domain_info = []
            healthcare_domains = []
            education_domains = []
            other_domains_list = []
            
            # Domain category mapping
            healthcare_keywords = ['health', 'medical', 'nutrition']
            education_keywords = ['academic', 'education', 'tutoring', 'learning']
            
            for domain in sorted(other_domains):
                try:
                    from app.rag.domain_retrievers import get_domain_retriever
                    retriever = get_domain_retriever(domain)
                    domain_stats = retriever.get_collection_stats()
                    chunk_count = domain_stats.get('chunk_count', 0)
                    doc_count = domain_stats.get('document_count', 0)
                except Exception:
                    # If domain can't be loaded, still include it without stats
                    chunk_count = 0
                    doc_count = 0
                
                domain_lower = domain.lower()
                if any(kw in domain_lower for kw in healthcare_keywords):
                    healthcare_domains.append((domain, chunk_count, doc_count))
                elif any(kw in domain_lower for kw in education_keywords):
                    education_domains.append((domain, chunk_count, doc_count))
                else:
                    other_domains_list.append((domain, chunk_count, doc_count))
            
            # Build markdown
            sections = [
                "## Alternatives and Related Datasets",
                "",
                "Looking for other domains? Check out the complete meeTARA Vectorstore collection:",
                ""
            ]
            
            # Healthcare section
            if healthcare_domains:
                sections.append("### Healthcare Domain")
                for domain, chunks, docs in healthcare_domains:
                    emoji = "🏥" if "general" in domain.lower() else "🧠" if "mental" in domain.lower() else "👩" if "women" in domain.lower() else "💊"
                    chunk_str = f"{chunks:,} chunks" if chunks > 0 else "Available"
                    sections.append(f"- {emoji} `{repo_prefix}-{domain}` - {domain.replace('_', ' ').title()} ({chunk_str})")
                sections.append("")
            
            # Education section
            if education_domains:
                sections.append("### Education Domain")
                for domain, chunks, docs in education_domains:
                    emoji = "📚"
                    chunk_str = f"{chunks:,} chunks" if chunks > 0 else "Available"
                    sections.append(f"- {emoji} `{repo_prefix}-{domain}` - {domain.replace('_', ' ').title()} ({chunk_str})")
                sections.append("")
            
            # Other domains section
            if other_domains_list:
                sections.append("### Other Domains")
                for domain, chunks, docs in other_domains_list:
                    emoji = "📁"
                    chunk_str = f"{chunks:,} chunks" if chunks > 0 else "Available"
                    sections.append(f"- {emoji} `{repo_prefix}-{domain}` - {domain.replace('_', ' ').title()} ({chunk_str})")
                sections.append("")
            
            sections.extend([
                f"**🔗 View all meeTARA datasets**: "
                f"[https://huggingface.co/{repo_prefix.split('/')[0]}](https://huggingface.co/{repo_prefix.split('/')[0]})",
                "",
                "**💡 Tip**: Combine multiple domain datasets for comprehensive multi-domain RAG applications!"
            ])
            
            return "\n".join(sections)
            
        except Exception as e:
            rag_logger.warning(f"Could not generate dynamic related datasets: {e}")
            # Fallback to simple link
            return (
                "## Alternatives and Related Datasets\n\n"
                "Looking for other domains? Check out the complete meeTARA Vectorstore collection!\n\n"
                f"**🔗 View all meeTARA datasets**: "
                f"[https://huggingface.co/{repo_prefix.split('/')[0]}](https://huggingface.co/{repo_prefix.split('/')[0]})\n"
            )
    
    def create_dataset_card(
        self,
        domain: str,
        stats: Dict[str, Any],
        embedding_model: str,
        repo_id: Optional[str] = None
    ) -> str:
        """Create a dataset card (README.md) for the vectorstore dataset.
        
        Args:
            domain: Domain name
            stats: Statistics from get_collection_stats()
            embedding_model: Embedding model used
            repo_id: Repository ID (e.g., "meetara-lab/vectorstore-general_health")
            
        Returns:
            Dataset card content as Markdown string
        """
        repo_id = repo_id or f"meetara-lab/vectorstore-{domain}"
        repo_prefix = '/'.join(repo_id.split('/')[:-1]) + '/vectorstore' if '/' in repo_id else "meetara-lab/vectorstore"
        
        # Get domain-specific examples
        domain_examples = self._get_domain_examples(domain)
        
        # Generate dynamic related datasets section
        related_datasets_section = self._get_related_datasets_section(domain, repo_prefix)
        
        # Determine size category based on actual chunk count
        chunk_count = stats.get('chunk_count', 0)
        if chunk_count < 1000:
            size_category = "n<1K"
        elif chunk_count < 10000:
            size_category = "1K<n<10K"
        elif chunk_count < 100000:
            size_category = "10K<n<100K"
        elif chunk_count < 1000000:
            size_category = "100K<n<1M"
        elif chunk_count < 10000000:
            size_category = "1M<n<10M"
        else:
            size_category = "10M<n<100M"
        
        # Create pretty name
        pretty_name = domain.replace('_', ' ').title()
        
        card = f"""---
license: apache-2.0
task_categories:
- feature-extraction
- text-retrieval
- question-answering
task_ids:
- semantic-similarity-scoring
- document-retrieval
- open-domain-qa
language:
- en
tags:
- embeddings
- vector-database
- rag
- retrieval-augmented-generation
- semantic-search
- knowledge-base
- {domain.replace('_', '-')}
size_categories:
- {size_category}
annotations_creators:
- machine-generated
language_creators:
- found
multilinguality: monolingual
pretty_name: {pretty_name} Vectorstore Dataset
source_datasets:
- original
---

# Vectorstore Dataset: {pretty_name}

## Overview

This dataset contains pre-computed vector embeddings for the **{domain.replace('_', ' ')}** domain, ready for use in Retrieval-Augmented Generation (RAG) applications, semantic search, and knowledge base systems. The embeddings are generated from high-quality source documents using state-of-the-art sentence transformers, making it easy to build production-ready RAG applications without the computational overhead of embedding generation.

## Key Features

- ✅ **Pre-computed embeddings**: Ready-to-use vector embeddings, saving computation time
- ✅ **Production-ready**: Optimized for real-world RAG applications
- ✅ **Comprehensive metadata**: Includes source file information, page numbers, and document hashes
- ✅ **LangChain compatible**: Works seamlessly with LangChain and ChromaDB
- ✅ **Search-optimized**: Designed for fast semantic similarity search

## What's Included

This dataset contains **{stats.get('chunk_count', 0):,}** text chunks from **{stats.get('document_count', 0):,}** source documents, each pre-embedded using the `{embedding_model}` model. Each chunk includes:
- **Text content**: The original document text
- **Embedding vector**: 384-dimensional float32 vector
- **Rich metadata**: Source file, page numbers, document hash, and more

## Dataset Details

### Dataset Summary

- **Domain**: `{domain}`
- **Total Chunks**: {stats.get('chunk_count', 0):,}
- **Total Documents**: {stats.get('document_count', 0):,}
- **Database Size**: {stats.get('db_size', 'Unknown')}
- **Embedding Model**: `{embedding_model}`
- **Chunk Size**: {stats.get('chunk_size', settings.chunk_size)}
- **Chunk Overlap**: {stats.get('chunk_overlap', settings.chunk_overlap)}

### Dataset Structure

The dataset contains the following columns:

- **id**: Unique identifier for each chunk
- **embedding**: Vector embedding (numpy array, dtype=float32)
- **document**: Original text content of the chunk
- **metadata**: JSON string containing metadata (file_name, file_hash, page_number, etc.)

### Embedding Model

This dataset uses embeddings from: `{embedding_model}`

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access the data
print(dataset["train"][0])
# Output:
# {{
#     'id': '...',
#     'embedding': array([...], dtype=float32),
#     'document': '...',
#     'metadata': '{{"file_name": "...", "page": 1, ...}}'
# }}
```

### Loading Back into ChromaDB

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("{repo_id}")["train"]

# Initialize ChromaDB
embeddings = HuggingFaceEmbeddings(model_name="{embedding_model}")
vectorstore = Chroma(
    persist_directory="./chroma_{domain}",
    embedding_function=embeddings
)

# Add documents to ChromaDB
documents = []
metadatas = []
ids = []
embeddings_list = []

for item in dataset:
    ids.append(item["id"])
    embeddings_list.append(item["embedding"].tolist())
    documents.append(item["document"])
    metadatas.append(json.loads(item["metadata"]))

# Note: You'll need to use ChromaDB's Python client directly for custom embeddings
import chromadb
client = chromadb.PersistentClient(path="./chroma_{domain}")
collection = client.create_collection(name="{domain}")

collection.add(
    ids=ids,
    embeddings=embeddings_list,
    documents=documents,
    metadatas=metadatas
)
```

### Using with LangChain

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize retriever
embeddings = HuggingFaceEmbeddings(model_name="{embedding_model}")
vectorstore = Chroma(
    persist_directory="./chroma_{domain}",
    embedding_function=embeddings
)

# Load from HF Hub first (see above), then use with LangChain
retriever = vectorstore.as_retriever()
results = retriever.invoke("your query here")
```

### Domain-Specific Usage Examples

This vectorstore is optimized for **{domain.replace('_', ' ').title()}** domain queries. Here are practical examples:

#### Example Queries

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load vectorstore (see "Loading Back into ChromaDB" above)
embeddings = HuggingFaceEmbeddings(model_name="{embedding_model}")
vectorstore = Chroma(
    persist_directory="./chroma_{domain}",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={{"k": 3}})

# Example queries for {domain.replace('_', ' ')} domain:
example_queries = {json.dumps(domain_examples["example_queries"], indent=4)}

# Run a query
query = "{domain_examples["example_queries"][0]}"
results = retriever.invoke(query)

# Display results
for i, doc in enumerate(results, 1):
    print(f"\\nResult {{i}}:")
    print(f"  Source: {{doc.metadata.get('file_name', 'Unknown')}}")
    print(f"  Page: {{doc.metadata.get('page', 'N/A')}}")
    print(f"  Content: {{doc.page_content[:200]}}...")
```

#### Common Use Cases

This dataset is useful for:
{chr(10).join(f"- **{use_case}**" for use_case in domain_examples["use_cases"])}

#### Real-World Example

```python
# Complete example: Query and use results
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Initialize (after loading from HF Hub)
embeddings = HuggingFaceEmbeddings(model_name="{embedding_model}")
vectorstore = Chroma(
    persist_directory="./chroma_{domain}",
    embedding_function=embeddings
)

# 2. Create retriever with relevance filtering
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={{
        "k": 5,  # Get top 5 most relevant results
        "score_threshold": 0.7  # Minimum similarity score
    }}
)

# 3. Query the vectorstore
query = "{domain_examples["example_queries"][1]}"
docs = retriever.invoke(query)

# 4. Process results
for doc in docs:
    metadata = doc.metadata
    print(f"📄 File: {{metadata.get('file_name', 'Unknown')}}")
    print(f"📃 Page: {{metadata.get('page', 'N/A')}}")
    print(f"📝 Content: {{doc.page_content[:300]}}...\\n")
```

## Dataset Statistics

### Content Statistics
- **Total Chunks**: {stats.get('chunk_count', 0):,}
- **Total Documents**: {stats.get('document_count', 0):,}
- **Average Chunks per Document**: {stats.get('chunk_count', 0) / max(1, stats.get('document_count', 1)):.1f}
- **Database Size**: {stats.get('db_size', 'Unknown')}

### Technical Specifications
- **Embedding Model**: `{embedding_model}` (384 dimensions)
- **Chunk Size**: {stats.get('chunk_size', settings.chunk_size)} characters
- **Chunk Overlap**: {stats.get('chunk_overlap', settings.chunk_overlap)} characters
- **Format**: Parquet/Arrow (optimized for fast loading)

## Performance Considerations

### Loading Time
- Full dataset loads in ~5-15 seconds on average hardware
- Memory usage: ~{round(stats.get('chunk_count', 0) * 384 * 4 / 1024 / 1024, 1)} MB for embeddings alone
- Recommended RAM: 2GB+ for full dataset operations

### Search Performance
- Typical query time: <100ms for similarity search
- Optimized for retrieval of top-k results (k=5-10)
- Works best with vector databases like ChromaDB, Pinecone, or Weaviate

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{meetara_vectorstore_{domain},
  title={{meeTARA Vectorstore: {domain.replace('_', ' ').title()}}},
  author={{meeTARA Lab}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_id}}}
}}
```

## Limitations and Considerations

- **Language**: This dataset is monolingual (English only)
- **Domain specificity**: Optimized for {domain.replace('_', ' ')} domain queries
- **Embedding model**: Uses `{embedding_model}` - ensure compatibility if switching models
- **Update frequency**: Dataset reflects state at time of publication; source documents may have been updated

{related_datasets_section}

## Maintenance and Updates

This dataset is maintained by the meeTARA Lab team. 

- **Last Updated**: {datetime.now().strftime("%Y-%m-%d")}
- **Version**: 1.0
- **Update Policy**: Datasets are updated periodically as source documents are added or improved
- **Notifications**: Follow the repository to receive updates when new versions are published

For updates, bug reports, or feature requests, please visit our GitHub repository.

## License

This dataset is released under the **Apache 2.0 License**. This means you are free to:
- Use the dataset commercially and non-commercially
- Modify and create derivative works
- Distribute the dataset and modifications

Please see the full license text for complete terms.

## Citation

If you use this dataset in your research or applications, please cite it as:

```bibtex
@dataset{{meetara_vectorstore_{domain},
  title={{meeTARA Vectorstore: {pretty_name}}},
  author={{meeTARA Lab}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{repo_id}}},
  license={{apache-2.0}},
  task={{feature-extraction, text-retrieval, rag}}
}}
```

## Contact and Support

- **GitHub**: [meetara-lab/meetara-core](https://github.com/meetara-lab/meetara-core)
- **Hugging Face Profile**: [@meetara-lab](https://huggingface.co/meetara-lab)
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/meetara-lab/meetara-core/issues)
- **Documentation**: Visit our repository for detailed documentation
- **Dataset Requests**: Want a new domain? [Open an issue](https://github.com/meetara-lab/meetara-core/issues) to request it!

## Contributing

We welcome contributions! If you'd like to:
- 🐛 Report bugs
- 💡 Suggest new domains
- 📝 Improve documentation
- 🔧 Contribute code

Please visit our [GitHub repository](https://github.com/meetara-lab/meetara-core).

---

**Made with ❤️ by the meeTARA Lab team**

**Part of the meeTARA Vectorstore Collection** - Empowering RAG applications with high-quality domain-specific embeddings.
"""
        return card
    
    def publish_domain(
        self,
        domain: str,
        repo_id: str,
        private: bool = False,
        create_pr: bool = False,
        commit_message: Optional[str] = None,
        push_to_hub: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Publish a domain's vectorstore to Hugging Face Hub.
        
        Args:
            domain: Domain name to publish
            repo_id: Repository ID (e.g., "meetara-lab/vectorstore-general_health")
            private: Whether to create a private repository
            create_pr: Whether to create a pull request instead of direct commit
            commit_message: Custom commit message
            push_to_hub: If False, only exports locally without uploading
            
        Returns:
            Tuple of (success: bool, repo_url: Optional[str])
        """
        try:
            # If no token, try to get from cache (from huggingface-cli login)
            if not self.token:
                try:
                    self.token = HfFolder.get_token()
                    if self.token:
                        rag_logger.info("Using token from Hugging Face login cache")
                        # Update API with cached token
                        self.api = HfApi(token=self.token)
                except Exception as e:
                    rag_logger.debug(f"Could not get token from cache: {e}")
            
            # Only warn if still no token - let huggingface_hub try its default loading
            if not self.token:
                rag_logger.warning(
                    "No explicit token provided. "
                    "Will attempt to use token from 'huggingface-cli login' or environment."
                )
            
            rag_logger.info(f"Publishing domain '{domain}' to HF Hub: {repo_id}")
            
            # Validate domain matches repo_id to prevent accidental mismatches
            repo_domain = repo_id.split('/')[-1].replace('vectorstore-', '') if 'vectorstore-' in repo_id else None
            if repo_domain and repo_domain != domain:
                rag_logger.warning(
                    f"WARNING: Domain mismatch detected! "
                    f"You are publishing domain '{domain}' to repository '{repo_id}'. "
                    f"The repository name suggests domain '{repo_domain}'. "
                    f"This may cause confusion. Please verify this is intentional."
                )
            
            # Export domain to dataset
            retriever = get_domain_retriever(domain)
            stats = retriever.get_collection_stats()
            
            # Validate that we got stats for the correct domain
            rag_logger.info(f"Stats for domain '{domain}': {stats.get('chunk_count', 0):,} chunks, {stats.get('document_count', 0):,} documents")
            
            # Export to dataset format (without saving to disk first)
            dataset = self.export_domain_to_dataset(
                domain=domain,
                output_dir=None,  # Don't save to disk - we'll push directly
                include_embeddings=True,
                include_documents=True,
                include_metadata=True
            )
            
            if not dataset:
                raise ValueError(f"Failed to export dataset for domain '{domain}'")
            
            # Create dataset card with validated domain and stats
            rag_logger.info(f"Creating dataset card for domain: '{domain}'")
            card_content = self.create_dataset_card(
                domain=domain,
                stats=stats,
                embedding_model=settings.embedding_model,
                repo_id=repo_id
            )
            
            # Verify card contains correct domain
            if f"**{domain}**" not in card_content and f"`{domain}`" not in card_content:
                rag_logger.warning(f"Dataset card may not contain correct domain '{domain}'")
            
            # Upload files using push_to_hub for better format handling
            rag_logger.debug(f"push_to_hub flag: {push_to_hub} (type: {type(push_to_hub)})")
            if push_to_hub:
                rag_logger.info("Entering upload block...")
                try:
                    create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=private,
                        token=self.token,
                        exist_ok=True
                    )
                    rag_logger.info(f"Repository ready: {repo_id}")
                    
                    rag_logger.info("Uploading dataset to Hugging Face Hub...")
                    
                    commit_msg = commit_message or (
                        f"Update vectorstore for domain: {domain}\n\n"
                        f"- Total chunks: {stats.get('chunk_count', 0):,}\n"
                        f"- Total documents: {stats.get('document_count', 0):,}\n"
                        f"- Embedding model: {settings.embedding_model}"
                    )
                    
                    # Delete old dataset files first
                    try:
                        existing_files = list_repo_files(
                            repo_id=repo_id,
                            repo_type="dataset",
                            token=self.token
                        )
                        # Delete old dataset files (Parquet/Arrow files) but keep README.md and .gitattributes
                        files_to_delete = [
                            f for f in existing_files 
                            if (f.endswith(('.parquet', '.arrow', '.json', '.arrow.schema')) 
                                or f.startswith('data-') or f.startswith('dataset_info'))
                            and not f.endswith('.md') and not f.startswith('.')
                        ]
                        if files_to_delete:
                            rag_logger.info(f"Deleting {len(files_to_delete)} old dataset files...")
                            for file_path in files_to_delete:
                                try:
                                    self.api.delete_file(
                                        path_in_repo=file_path,
                                        repo_id=repo_id,
                                        repo_type="dataset",
                                        token=self.token,
                                        commit_message=f"Delete old {file_path} before republish"
                                    )
                                except Exception as e:
                                    rag_logger.warning(f"  Could not delete {file_path}: {e}")
                    except Exception as e:
                        rag_logger.debug(f"Could not list/delete old files (new repo?): {e}")
                    
                    # Use push_to_hub directly on Dataset - this handles format state better
                    push_kwargs = {
                        "repo_id": repo_id,
                        "token": self.token,
                        "commit_message": commit_msg,
                    }
                    
                    # Only add private if creating a new repo (repo doesn't exist yet)
                    try:
                        existing_files = list_repo_files(
                            repo_id=repo_id,
                            repo_type="dataset",
                            token=self.token
                        )
                        repo_exists = len(existing_files) > 0
                    except Exception:
                        repo_exists = False
                    
                    if not repo_exists:
                        push_kwargs["private"] = private
                    
                    if create_pr:
                        push_kwargs["create_pr"] = True
                    
                    rag_logger.info(f"Pushing dataset to {repo_id}...")
                    rag_logger.info(f"Dataset info: {len(dataset)} rows, features: {list(dataset.features.keys())}")
                    
                    # Verify dataset is valid before pushing
                    if len(dataset) == 0:
                        raise ValueError("Cannot push empty dataset to Hub")
                    
                    # Push dataset - this uploads the actual data files
                    try:
                        dataset.push_to_hub(**push_kwargs)
                        rag_logger.info("Dataset pushed to Hub successfully")
                        rag_logger.info(f"Uploaded {len(dataset):,} rows with {len(dataset.features)} features")
                    except Exception as push_error:
                        rag_logger.error(f"Error pushing dataset: {push_error}", exc_info=True)
                        raise ValueError(f"Failed to push dataset to Hugging Face Hub: {str(push_error)}") from push_error
                    
                    # Upload README.md separately
                    try:
                        from huggingface_hub import upload_file
                        upload_file(
                            path_or_fileobj=card_content.encode('utf-8'),
                            path_in_repo="README.md",
                            repo_id=repo_id,
                            repo_type="dataset",
                            token=self.token,
                            commit_message="Update dataset card",
                        )
                        rag_logger.info("README.md uploaded")
                    except Exception as e:
                        rag_logger.warning(f"Could not upload README.md separately: {e}")
                    
                    repo_url = f"https://huggingface.co/datasets/{repo_id}"
                    rag_logger.info(f"Successfully published to: {repo_url}")
                    
                    return True, repo_url
                except Exception as e:
                    rag_logger.error(f"Error during upload: {e}", exc_info=True)
                    rag_logger.error(f"Exception type: {type(e).__name__}")
                    raise ValueError(f"Failed to upload dataset to Hugging Face Hub: {str(e)}") from e
            else:
                rag_logger.warning(f"push_to_hub is False - skipping upload")
                rag_logger.info(f"Dataset exported locally (push_to_hub=False)")
                return True, None
                    
        except Exception as e:
            rag_logger.error(f"Failed to publish domain '{domain}': {e}", exc_info=True)
            rag_logger.error(f"Exception type: {type(e).__name__}")
            return False, str(e)
    
    def publish_all_domains(
        self,
        repo_prefix: str = "meetara-lab/vectorstore",
        private: bool = False,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Publish all available domains to Hugging Face Hub.
        
        Args:
            repo_prefix: Prefix for repository IDs (e.g., "meetara-lab/vectorstore")
            private: Whether to create private repositories
            domains: List of domains to publish. If None, publishes all available domains.
            
        Returns:
            Dictionary mapping domain names to (success, repo_url) tuples
        """
        if domains is None:
            domains = list_available_domains()
        
        results = {}
        
        rag_logger.info(f"Publishing {len(domains)} domains to HF Hub...")
        
        for domain in domains:
            repo_id = f"{repo_prefix}-{domain}"
            success, repo_url = self.publish_domain(
                domain=domain,
                repo_id=repo_id,
                private=private
            )
            results[domain] = (success, repo_url)
            
            if success:
                rag_logger.info(f"{domain}: {repo_url or 'Exported locally'}")
            else:
                rag_logger.error(f"{domain}: Failed to publish")
        
        return results


def load_vectorstore_from_hub(
    repo_id: str,
    domain: str,
    local_dir: Optional[Path] = None,
    embedding_model: Optional[str] = None
):
    """Load a vectorstore dataset from Hugging Face Hub back into ChromaDB.
    
    Args:
        repo_id: Repository ID (e.g., "meetara-lab/vectorstore-general_health")
        domain: Domain name for the vectorstore
        local_dir: Local directory to save ChromaDB files. If None, uses default.
        embedding_model: Embedding model to use. If None, uses settings default.
        
    Returns:
        ChromaDB vectorstore instance
    """
    try:
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )
        
        rag_logger.info(f"Loading vectorstore from HF Hub: {repo_id}")
        
        # Load dataset
        dataset = load_dataset(repo_id, split="train")
        
        rag_logger.info(f"Loaded {len(dataset):,} chunks from dataset")
        
        # Initialize ChromaDB
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        if local_dir is None:
            local_dir = settings.vectorstore_path / domain
        else:
            local_dir = Path(local_dir)
        
        embedding_model = embedding_model or settings.embedding_model
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize ChromaDB client directly for custom embeddings
        import chromadb
        
        local_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(local_dir))
        
        # Create or get collection
        try:
            collection = client.get_collection(name="langchain")
        except:
            collection = client.create_collection(
                name="langchain",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Prepare data
        ids = []
        embeddings_list = []
        documents = []
        metadatas = []
        
        for item in dataset:
            ids.append(item["id"])
            if "embedding" in item:
                emb = item["embedding"]
                if isinstance(emb, np.ndarray):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(emb)
            documents.append(item.get("document", ""))
            if "metadata" in item:
                meta = item["metadata"]
                if isinstance(meta, str):
                    metadatas.append(json.loads(meta))
                else:
                    metadatas.append(meta)
            else:
                metadatas.append({})
        
        # Add to collection in batches
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings_list[i:i+batch_size]
            batch_documents = documents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
            
            if (i // batch_size + 1) % 10 == 0:
                rag_logger.info(f"Processed {i + len(batch_ids):,}/{len(ids):,} chunks")
        
        # Create LangChain wrapper
        vectorstore = Chroma(
            persist_directory=str(local_dir),
            embedding_function=embeddings
        )
        
        rag_logger.info(f"Successfully loaded {len(ids):,} chunks into ChromaDB")
        
        return vectorstore
        
    except Exception as e:
        rag_logger.error(f"Failed to load vectorstore from Hub: {e}", exc_info=True)
        raise

