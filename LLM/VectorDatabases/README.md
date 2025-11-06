# VectorDatabases

**Production-Ready Vector Database Management System**

A comprehensive, enterprise-grade vector database system with support for multiple backends, advanced search capabilities, and production-ready features.

**Author:** BrillConsulting
**Version:** 1.0.0
**License:** MIT

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Backends](#supported-backends)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
  - [Basic Operations](#basic-operations)
  - [CRUD Operations](#crud-operations)
  - [Similarity Search](#similarity-search)
  - [Filtering](#filtering)
  - [Batch Operations](#batch-operations)
  - [Hybrid Search](#hybrid-search)
  - [Index Optimization](#index-optimization)
- [Backend Configuration](#backend-configuration)
- [Advanced Features](#advanced-features)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

VectorDatabases is a unified, production-ready system for managing vector embeddings across multiple database backends. It provides a consistent API for vector storage, retrieval, and similarity search, making it easy to switch between different vector database technologies without code changes.

### Key Benefits

- **Backend Agnostic**: Single API works with FAISS, ChromaDB, Pinecone, Weaviate, and Qdrant
- **Production Ready**: Comprehensive error handling, logging, and performance optimization
- **Hybrid Search**: Combines vector similarity with keyword matching for improved results
- **Scalable**: Supports batch operations and index optimization for large-scale deployments
- **Type Safe**: Full type hints and dataclass-based document models
- **Flexible**: Configurable distance metrics, index types, and search parameters

---

## Features

### Core Functionality

- **Multiple Backend Support**: FAISS, ChromaDB, Pinecone, Weaviate, Qdrant
- **CRUD Operations**: Complete Create, Read, Update, Delete functionality
- **Similarity Search**: Fast vector similarity search with multiple distance metrics
- **Metadata Filtering**: Filter search results by document metadata
- **Batch Operations**: Efficient bulk insert and delete operations
- **Hybrid Search**: Combine vector and keyword search for better results
- **Index Optimization**: Backend-specific optimization for improved performance
- **Import/Export**: Save and load indexes for persistence

### Advanced Features

- **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
- **Multiple Index Types**: Flat, IVF, HNSW, LSH
- **Configurable Index Parameters**: Fine-tune performance characteristics
- **Statistics and Monitoring**: Track index size, vector count, and performance
- **Logging**: Comprehensive logging for debugging and monitoring
- **Error Handling**: Robust error handling with detailed error messages

---

## Supported Backends

| Backend | Description | Best For | Cloud/Local |
|---------|-------------|----------|-------------|
| **FAISS** | Facebook AI Similarity Search | Fast local similarity search, prototyping | Local |
| **ChromaDB** | Open-source embedding database | Development, small-medium scale | Local |
| **Pinecone** | Managed vector database | Production deployments at scale | Cloud |
| **Weaviate** | Open-source vector search engine | Complex schemas, GraphQL API | Both |
| **Qdrant** | High-performance vector search | High-throughput applications | Both |

---

## Installation

### Basic Installation

```bash
# Install core dependencies
pip install numpy>=1.24.0

# Install specific backend (choose one or more)
pip install faiss-cpu>=1.7.4              # FAISS (CPU)
pip install faiss-gpu>=1.7.4              # FAISS (GPU)
pip install chromadb>=0.4.0               # ChromaDB
pip install pinecone-client>=2.2.0        # Pinecone
pip install weaviate-client>=3.24.0       # Weaviate
pip install qdrant-client>=1.6.0          # Qdrant
```

### Install All Backends

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Basic Example

```python
from vector_databases import VectorDatabaseManager, VectorDocument, IndexConfig
import numpy as np

# Initialize manager with FAISS backend
manager = VectorDatabaseManager(backend_type="faiss")

# Create a document
doc = VectorDocument(
    id="doc_1",
    vector=np.random.randn(768).astype(np.float32),
    text="This is a sample document",
    metadata={"category": "example", "priority": 1}
)

# Insert document
manager.insert(doc)

# Search for similar vectors
query_vector = np.random.randn(768).astype(np.float32)
results = manager.search(query_vector, top_k=5)

# Print results
for result in results:
    print(f"ID: {result.id}, Score: {result.score:.4f}")
```

### 30-Second Demo

```python
from vector_databases import demo_faiss, demo_hybrid_search

# Run FAISS demo
demo_faiss()

# Run hybrid search demo
demo_hybrid_search()
```

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────┐
│     VectorDatabaseManager (Facade)      │
│  - Unified API across all backends      │
│  - Hybrid search support                │
│  - Import/export functionality          │
└──────────────┬──────────────────────────┘
               │
               ├───────┬────────┬────────┬────────┐
               │       │        │        │        │
         ┌─────▼──┐ ┌──▼───┐ ┌─▼────┐ ┌─▼──────┐ ┌─▼──────┐
         │ FAISS  │ │Chroma│ │Pinec.│ │Weaviate│ │ Qdrant │
         │Backend │ │Backend│ │Backend│ │Backend │ │Backend │
         └────────┘ └──────┘ └──────┘ └────────┘ └────────┘
```

### Core Classes

- **`VectorDatabaseManager`**: Main facade providing unified API
- **`VectorDatabaseBackend`**: Abstract base class for all backends
- **`VectorDocument`**: Document model with vector, text, and metadata
- **`SearchResult`**: Search result with score and ranking
- **`IndexConfig`**: Configuration for index creation
- **`HybridSearchEngine`**: Combines vector and keyword search

---

## Usage Guide

### Basic Operations

#### Initialize Manager

```python
from vector_databases import VectorDatabaseManager, IndexConfig, IndexType, DistanceMetric

# Basic initialization
manager = VectorDatabaseManager(backend_type="faiss")

# Advanced configuration
config = IndexConfig(
    index_type=IndexType.HNSW,
    dimension=768,
    metric=DistanceMetric.COSINE,
    m=32,
    ef_construction=400
)
manager = VectorDatabaseManager(backend_type="faiss", config=config)
```

#### Backend-Specific Initialization

```python
# ChromaDB
manager = VectorDatabaseManager(
    backend_type="chroma",
    collection_name="my_collection"
)

# Pinecone
manager = VectorDatabaseManager(
    backend_type="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="my_index"
)

# Weaviate
manager = VectorDatabaseManager(
    backend_type="weaviate",
    url="http://localhost:8080",
    class_name="Document"
)

# Qdrant
manager = VectorDatabaseManager(
    backend_type="qdrant",
    url="http://localhost:6333",
    collection_name="my_collection"
)
```

### CRUD Operations

#### Create (Insert)

```python
from vector_databases import VectorDocument
import numpy as np

# Create document
doc = VectorDocument(
    id="doc_1",
    vector=np.random.randn(768).astype(np.float32),
    text="Document text content",
    metadata={"category": "technical", "author": "John Doe"}
)

# Insert single document
success = manager.insert(doc)
print(f"Insert successful: {success}")
```

#### Read (Get)

```python
# Retrieve document by ID
document = manager.get("doc_1")
if document:
    print(f"Found: {document.text}")
    print(f"Metadata: {document.metadata}")
```

#### Update

```python
# Update document
doc.text = "Updated text content"
doc.metadata["version"] = 2
success = manager.update("doc_1", doc)
print(f"Update successful: {success}")
```

#### Delete

```python
# Delete single document
success = manager.delete("doc_1")
print(f"Delete successful: {success}")
```

### Similarity Search

#### Basic Search

```python
# Create query vector
query_vector = np.random.randn(768).astype(np.float32)

# Search for top 10 similar documents
results = manager.search(query_vector, top_k=10)

for result in results:
    print(f"Rank {result.rank}: {result.id}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Text: {result.document.text[:100]}")
```

#### Search with Distance Metrics

```python
from vector_databases import DistanceMetric

# Configure different distance metrics
config = IndexConfig(
    dimension=768,
    metric=DistanceMetric.EUCLIDEAN  # or COSINE, DOT_PRODUCT, MANHATTAN
)
manager = VectorDatabaseManager(backend_type="faiss", config=config)
```

### Filtering

#### Metadata Filtering

```python
# Search with metadata filters
results = manager.search(
    query_vector,
    top_k=10,
    filters={"category": "technical", "priority": 1}
)

# Multiple filter conditions
results = manager.search(
    query_vector,
    top_k=10,
    filters={
        "author": "John Doe",
        "status": "published",
        "year": 2024
    }
)
```

### Batch Operations

#### Batch Insert

```python
from vector_databases import create_sample_documents

# Create multiple documents
documents = create_sample_documents(num_docs=1000, dimension=768)

# Batch insert
result = manager.batch_insert(documents)
print(f"Inserted: {result['inserted']}")
print(f"Total vectors: {result['total']}")
```

#### Batch Delete

```python
# Delete multiple documents
doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
result = manager.batch_delete(doc_ids)
print(f"Deleted: {result['deleted']}")
```

### Hybrid Search

Hybrid search combines vector similarity with keyword matching for improved search quality.

```python
# Enable hybrid search
manager.enable_hybrid_search(alpha=0.7)  # 70% vector, 30% keyword

# Perform hybrid search
results = manager.search(
    query_vector=query_vector,
    query_text="machine learning artificial intelligence",
    top_k=10
)

for result in results:
    print(f"{result.id}: {result.score:.4f}")
```

#### Adjusting Hybrid Search Weights

```python
# More weight on vector similarity
manager.enable_hybrid_search(alpha=0.9)  # 90% vector, 10% keyword

# Balanced
manager.enable_hybrid_search(alpha=0.5)  # 50% vector, 50% keyword

# More weight on keywords
manager.enable_hybrid_search(alpha=0.3)  # 30% vector, 70% keyword
```

### Index Optimization

```python
# Optimize index for better performance
result = manager.optimize_index()
print(f"Optimization: {result}")

# Get index statistics
stats = manager.get_stats()
print(f"Backend: {stats['backend']}")
print(f"Total vectors: {stats['total_vectors']}")
print(f"Dimension: {stats['dimension']}")
print(f"Metric: {stats['metric']}")
```

---

## Backend Configuration

### FAISS Configuration

```python
from vector_databases import IndexType, DistanceMetric

# Flat index (exact search)
config = IndexConfig(
    index_type=IndexType.FLAT,
    dimension=768,
    metric=DistanceMetric.COSINE
)

# IVF index (faster approximate search)
config = IndexConfig(
    index_type=IndexType.IVF,
    dimension=768,
    metric=DistanceMetric.COSINE,
    nlist=100  # Number of clusters
)

# HNSW index (hierarchical navigable small world)
config = IndexConfig(
    index_type=IndexType.HNSW,
    dimension=768,
    metric=DistanceMetric.COSINE,
    m=16,                    # Number of connections
    ef_construction=200,      # Construction time accuracy
    ef_search=50             # Search time accuracy
)
```

### ChromaDB Configuration

```python
manager = VectorDatabaseManager(
    backend_type="chroma",
    collection_name="my_collection"
)
```

### Pinecone Configuration

```python
manager = VectorDatabaseManager(
    backend_type="pinecone",
    api_key="your-api-key",
    environment="us-west1-gcp",
    index_name="my_index"
)
```

### Weaviate Configuration

```python
manager = VectorDatabaseManager(
    backend_type="weaviate",
    url="http://localhost:8080",
    class_name="Document"
)
```

### Qdrant Configuration

```python
manager = VectorDatabaseManager(
    backend_type="qdrant",
    url="http://localhost:6333",
    collection_name="my_collection"
)
```

---

## Advanced Features

### Import/Export Index

```python
# Export index to file
success = manager.export_index("/path/to/index.pkl")
print(f"Export successful: {success}")

# Import index from file
success = manager.import_index("/path/to/index.pkl")
print(f"Import successful: {success}")
```

### Custom Document Creation

```python
from datetime import datetime

doc = VectorDocument(
    id="custom_doc_1",
    vector=embedding_vector,
    text="Custom document with detailed metadata",
    metadata={
        "title": "Research Paper",
        "author": "Jane Smith",
        "date": datetime.now().isoformat(),
        "tags": ["AI", "ML", "NLP"],
        "version": 1.0,
        "department": "R&D"
    },
    timestamp=datetime.now().isoformat()
)
```

### Working with Search Results

```python
results = manager.search(query_vector, top_k=10)

# Convert to dictionary
for result in results:
    result_dict = result.to_dict()
    print(result_dict)

# Access document details
for result in results:
    doc = result.document
    print(f"ID: {doc.id}")
    print(f"Text: {doc.text}")
    print(f"Metadata: {doc.metadata}")
    print(f"Timestamp: {doc.timestamp}")
```

---

## Performance Considerations

### Choosing the Right Backend

- **FAISS**: Best for local development and CPU-based similarity search
- **ChromaDB**: Good for small to medium datasets with rich metadata
- **Pinecone**: Ideal for production deployments requiring scalability
- **Weaviate**: Best when you need GraphQL API and complex filtering
- **Qdrant**: Optimal for high-throughput applications

### Index Types and Performance

| Index Type | Search Speed | Memory | Accuracy | Best For |
|------------|--------------|--------|----------|----------|
| Flat | Slow | High | 100% | Small datasets (<10K) |
| IVF | Fast | Medium | ~95% | Medium datasets (10K-1M) |
| HNSW | Very Fast | High | ~99% | Large datasets (>1M) |
| LSH | Fast | Low | ~90% | Very large datasets |

### Batch Operations

Always use batch operations for bulk inserts/deletes:

```python
# Good: Batch insert
manager.batch_insert(documents)

# Bad: Individual inserts in loop
for doc in documents:
    manager.insert(doc)  # Much slower!
```

### Vector Normalization

Normalize vectors when using cosine similarity:

```python
# Normalize vectors
vector = vector / np.linalg.norm(vector)
```

---

## Best Practices

### 1. Choose Appropriate Dimension

- 384: Sentence transformers (lightweight)
- 768: BERT-based models (balanced)
- 1536: OpenAI embeddings
- 3072: Large language models

### 2. Use Metadata Wisely

```python
# Good: Structured, queryable metadata
metadata = {
    "category": "article",
    "date": "2024-01-15",
    "author": "John Doe",
    "status": "published"
}

# Bad: Unstructured, hard to query
metadata = {
    "info": "article by John Doe published on 2024-01-15"
}
```

### 3. Implement Error Handling

```python
try:
    results = manager.search(query_vector, top_k=10)
except Exception as e:
    logger.error(f"Search failed: {str(e)}")
    # Implement fallback logic
```

### 4. Monitor Performance

```python
import time

start = time.time()
results = manager.search(query_vector, top_k=10)
elapsed = time.time() - start

print(f"Search took {elapsed:.3f} seconds")
```

### 5. Regular Optimization

```python
# Optimize index periodically
if manager.get_stats()['total_vectors'] % 10000 == 0:
    manager.optimize_index()
```

---

## API Reference

### VectorDatabaseManager

**Constructor**
```python
VectorDatabaseManager(
    backend_type: str = "faiss",
    config: Optional[IndexConfig] = None,
    **backend_kwargs
)
```

**Methods**
- `insert(document: VectorDocument) -> bool`
- `batch_insert(documents: List[VectorDocument]) -> Dict[str, Any]`
- `search(query_vector, top_k, filters, query_text) -> List[SearchResult]`
- `get(doc_id: str) -> Optional[VectorDocument]`
- `update(doc_id: str, document: VectorDocument) -> bool`
- `delete(doc_id: str) -> bool`
- `batch_delete(doc_ids: List[str]) -> Dict[str, Any]`
- `optimize_index() -> Dict[str, Any]`
- `get_stats() -> Dict[str, Any]`
- `export_index(filepath: str) -> bool`
- `import_index(filepath: str) -> bool`
- `enable_hybrid_search(alpha: float = 0.5)`

### VectorDocument

**Attributes**
- `id: str` - Unique document identifier
- `vector: np.ndarray` - Embedding vector
- `metadata: Dict[str, Any]` - Document metadata
- `text: Optional[str]` - Document text content
- `timestamp: str` - Creation timestamp

**Methods**
- `to_dict() -> Dict[str, Any]`

### SearchResult

**Attributes**
- `id: str` - Document ID
- `score: float` - Similarity score
- `document: VectorDocument` - Full document
- `rank: int` - Result ranking

**Methods**
- `to_dict() -> Dict[str, Any]`

### IndexConfig

**Attributes**
- `index_type: IndexType` - Type of index
- `dimension: int` - Vector dimension
- `metric: DistanceMetric` - Distance metric
- `nlist: int` - IVF parameter
- `m: int` - HNSW parameter
- `ef_construction: int` - HNSW parameter
- `ef_search: int` - HNSW parameter

---

## Examples

### Example 1: Semantic Search System

```python
from vector_databases import VectorDatabaseManager, VectorDocument
import numpy as np

# Initialize
manager = VectorDatabaseManager(backend_type="faiss")

# Add documents
documents = [
    VectorDocument(
        id="1",
        vector=np.random.randn(768).astype(np.float32),
        text="Python programming tutorial",
        metadata={"type": "tutorial"}
    ),
    VectorDocument(
        id="2",
        vector=np.random.randn(768).astype(np.float32),
        text="Machine learning guide",
        metadata={"type": "guide"}
    )
]

manager.batch_insert(documents)

# Search
query = np.random.randn(768).astype(np.float32)
results = manager.search(query, top_k=5)
```

### Example 2: Document Management System

```python
# Initialize with ChromaDB
manager = VectorDatabaseManager(
    backend_type="chroma",
    collection_name="documents"
)

# Add documents with rich metadata
doc = VectorDocument(
    id="doc_2024_001",
    vector=embedding_model.encode(text),
    text=document_text,
    metadata={
        "title": "Q4 Report",
        "department": "Finance",
        "quarter": "Q4",
        "year": 2024,
        "confidential": False
    }
)

manager.insert(doc)

# Search with filters
results = manager.search(
    query_vector,
    top_k=10,
    filters={"department": "Finance", "year": 2024}
)
```

### Example 3: Multi-Backend Deployment

```python
# Development: Use FAISS
dev_manager = VectorDatabaseManager(backend_type="faiss")

# Production: Use Pinecone
prod_manager = VectorDatabaseManager(
    backend_type="pinecone",
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
    index_name="production"
)

# Same API for both!
dev_manager.insert(doc)
prod_manager.insert(doc)
```

---

## Troubleshooting

### Common Issues

**Issue: ImportError for backend**
```
Solution: Install the required backend
pip install faiss-cpu  # or chromadb, pinecone-client, etc.
```

**Issue: Dimension mismatch**
```
Solution: Ensure all vectors have the same dimension as IndexConfig
```

**Issue: Slow search performance**
```
Solution:
1. Use appropriate index type (HNSW for large datasets)
2. Optimize index regularly
3. Consider using approximate search
```

**Issue: Out of memory**
```
Solution:
1. Use IVF index instead of Flat
2. Process in smaller batches
3. Increase system RAM or use cloud backend
```

### Logging

Enable detailed logging for debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vector_databases")
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

---

## License

MIT License - see LICENSE file for details

---

## Support

For issues, questions, or contributions:
- **Email**: support@brillconsulting.com
- **GitHub**: [BrillConsulting](https://github.com/BrillConsulting)
- **Documentation**: See this README

---

## Changelog

### Version 1.0.0 (2025-11-06)
- Initial production release
- Support for 5 vector database backends
- CRUD operations
- Similarity search with multiple metrics
- Metadata filtering
- Batch operations
- Hybrid search capability
- Index optimization
- Import/export functionality
- Comprehensive documentation

---

**Built with excellence by BrillConsulting**
