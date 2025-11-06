# Production-Ready RAG System

A comprehensive, enterprise-grade Retrieval-Augmented Generation (RAG) system combining advanced semantic search with large language model generation for accurate, source-grounded responses.

## Overview

This RAG system provides a complete solution for building intelligent document Q&A applications with state-of-the-art retrieval and generation capabilities. It's designed for production use with features like hybrid search, reranking, citation tracking, and streaming responses.

## Key Features

### Multi-Model Embedding Support
- **OpenAI Embeddings**: GPT-powered embeddings (text-embedding-3-small/large)
- **HuggingFace Transformers**: Open-source sentence transformers (all-MiniLM-L6-v2, all-mpnet-base-v2)
- **Cohere Embeddings**: Multilingual embedding models (embed-english-v3.0)
- Extensible architecture for adding custom embedding models

### Advanced Chunking Strategies
- **Character-based Chunking**: Simple overlap-based splitting for general use
- **Sentence-aware Chunking**: Respects sentence boundaries for better context
- **Recursive Chunking**: Multi-level splitting with intelligent separators
- **Semantic Chunking**: Groups content by semantic similarity for coherent chunks
- Precise character-level position tracking for accurate citations

### Hybrid Search
- **Semantic Search**: Dense vector embeddings with cosine similarity
- **Keyword Search**: BM25 algorithm for exact term matching
- **Hybrid Fusion**: Configurable weighted combination (default: 70% semantic, 30% keyword)
- Handles both conceptual queries and specific terminology

### Intelligent Reranking
- Cross-encoder models for improved relevance scoring
- Two-stage retrieval pipeline: initial retrieval + reranking
- Significantly improves top-k precision

### Query Enhancement
- **Query Expansion**: Multiple query variations for better recall
- **Synonym Expansion**: Broadens search with related terms
- **Multi-query Generation**: Creates diverse query formulations
- **HyDE Support**: Hypothetical Document Embeddings for zero-shot scenarios

### Citation & Attribution
- Precise source tracking with character-level positions
- Confidence scores for each citation
- Automatic citation extraction from generated answers
- Multi-document synthesis with clear source attribution

### Streaming Responses
- Real-time response generation
- Progressive result delivery
- Better UX for long-form answers
- Compatible with modern streaming APIs

### Production Features
- Batch processing for efficient embedding generation
- Index persistence (save/load)
- Evaluation metrics
- Comprehensive error handling
- Type hints throughout
- Extensive documentation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      RAG System                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Embedding   │  │   Chunking   │  │   Search     │ │
│  │   Models     │  │  Strategies  │  │   Engine     │ │
│  │              │  │              │  │              │ │
│  │ • OpenAI     │  │ • Character  │  │ • Semantic   │ │
│  │ • HuggingFace│  │ • Sentence   │  │ • Keyword    │ │
│  │ • Cohere     │  │ • Recursive  │  │ • Hybrid     │ │
│  │              │  │ • Semantic   │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Reranker    │  │    Query     │  │  Citation    │ │
│  │              │  │  Expansion   │  │  Tracker     │ │
│  │ • Cross-enc. │  │              │  │              │ │
│  │ • Scoring    │  │ • Synonyms   │  │ • Extraction │ │
│  │              │  │ • Multi-Q    │  │ • Confidence │ │
│  │              │  │ • HyDE       │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Answer Synthesis & Generation             │  │
│  │  • Multi-document context building                │  │
│  │  • LLM integration (OpenAI, Anthropic, etc.)     │  │
│  │  • Streaming support                              │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional: API Keys

For production use with external services:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Cohere
export COHERE_API_KEY="your-api-key"
```

## Quick Start

### Basic Usage

```python
from rag_system import RAGSystem, HuggingFaceEmbedding, SentenceChunking

# Initialize system
rag = RAGSystem(
    embedding_model=HuggingFaceEmbedding(),
    chunking_strategy=SentenceChunking(chunk_size=500),
    enable_reranking=True,
    enable_query_expansion=True
)

# Add documents
rag.add_document(
    text="Machine learning is a subset of AI...",
    metadata={"title": "ML Basics", "author": "John Doe", "year": 2024}
)

# Process documents (chunk + embed)
rag.process_documents()

# Query the system
result = rag.query("What is machine learning?", top_k=5)

print(result["answer"])
print(f"Sources: {result['num_sources']}")
print(f"Citations: {len(result['citations'])}")
```

### Advanced Usage

```python
from rag_system import (
    RAGSystem,
    OpenAIEmbedding,
    SemanticChunking,
    CharacterChunking
)

# Custom configuration
rag = RAGSystem(
    embedding_model=OpenAIEmbedding(model="text-embedding-3-large"),
    chunking_strategy=SemanticChunking(
        embedding_model=HuggingFaceEmbedding(),
        similarity_threshold=0.75,
        max_chunk_size=1000
    ),
    enable_reranking=True,
    enable_query_expansion=True,
    hybrid_search_alpha=0.8  # 80% semantic, 20% keyword
)

# Batch document ingestion
documents = [
    {"text": doc1_text, "metadata": {"source": "paper1.pdf"}},
    {"text": doc2_text, "metadata": {"source": "paper2.pdf"}},
]

for doc in documents:
    rag.add_document(doc["text"], doc["metadata"])

# Process with batch optimization
rag.process_documents(batch_size=64)

# Advanced search with mode selection
results = rag.search(
    query="machine learning applications",
    top_k=10,
    search_mode="hybrid"  # or "semantic" or "keyword"
)

for result in results:
    print(f"Score: {result.combined_score:.3f}")
    print(f"Text: {result.chunk.text[:100]}...")
    print(f"Source: {result.chunk.metadata}")
```

### Streaming Responses

```python
# Stream responses for better UX
for chunk in rag.query_stream("Explain deep learning", top_k=5):
    if chunk["type"] == "search_complete":
        print(f"Found {chunk['num_results']} relevant chunks")

    elif chunk["type"] == "answer_chunk":
        print(chunk["content"], end="", flush=True)

    elif chunk["type"] == "complete":
        print(f"\n\nCitations: {len(chunk['citations'])}")
        print(f"Sources: {chunk['num_sources']}")
```

## API Reference

### RAGSystem Class

#### Initialization

```python
RAGSystem(
    embedding_model: Optional[EmbeddingModel] = None,
    chunking_strategy: Optional[ChunkingStrategy] = None,
    enable_reranking: bool = True,
    enable_query_expansion: bool = True,
    hybrid_search_alpha: float = 0.7
)
```

**Parameters:**
- `embedding_model`: Embedding model instance (default: HuggingFaceEmbedding)
- `chunking_strategy`: Chunking strategy instance (default: SentenceChunking)
- `enable_reranking`: Enable cross-encoder reranking (default: True)
- `enable_query_expansion`: Enable query expansion (default: True)
- `hybrid_search_alpha`: Weight for semantic vs keyword search (default: 0.7)

#### Methods

##### `add_document(text, metadata=None, doc_id=None)`
Add a document to the knowledge base.

**Parameters:**
- `text` (str): Document text
- `metadata` (dict, optional): Document metadata
- `doc_id` (str, optional): Custom document ID

**Returns:** Document ID (str)

##### `process_documents(batch_size=32)`
Process all documents: chunk and generate embeddings.

**Parameters:**
- `batch_size` (int): Batch size for embedding generation

##### `search(query, top_k=5, search_mode="hybrid")`
Search for relevant chunks.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return
- `search_mode` (str): "semantic", "keyword", or "hybrid"

**Returns:** List of SearchResult objects

##### `query(query, top_k=5, search_mode="hybrid")`
End-to-end RAG query with answer generation.

**Parameters:**
- `query` (str): User question
- `top_k` (int): Number of chunks to retrieve
- `search_mode` (str): Search mode

**Returns:** Dictionary with answer, citations, and metadata

##### `query_stream(query, top_k=5)`
Stream response generation.

**Parameters:**
- `query` (str): User question
- `top_k` (int): Number of chunks to retrieve

**Yields:** Dictionary chunks with streaming data

##### `save_index(filepath)`
Save the document index and embeddings.

**Parameters:**
- `filepath` (str): Path to save index

##### `load_index(filepath)`
Load document index and embeddings.

**Parameters:**
- `filepath` (str): Path to index file

##### `evaluate(test_queries, top_k=5)`
Evaluate system performance.

**Parameters:**
- `test_queries` (List[Tuple[str, str]]): List of (query, expected_answer) tuples
- `top_k` (int): Number of chunks to retrieve

**Returns:** Dictionary with evaluation metrics

### Embedding Models

#### OpenAIEmbedding
```python
OpenAIEmbedding(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
    api_key=None
)
```

#### HuggingFaceEmbedding
```python
HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"  # or "all-mpnet-base-v2", etc.
)
```

#### CohereEmbedding
```python
CohereEmbedding(
    model="embed-english-v3.0",
    api_key=None
)
```

### Chunking Strategies

#### CharacterChunking
Simple character-based chunking with overlap.
```python
CharacterChunking(chunk_size=500, overlap=50)
```

#### SentenceChunking
Respects sentence boundaries.
```python
SentenceChunking(chunk_size=500, overlap_sentences=2)
```

#### RecursiveChunking
Multi-level splitting with separators.
```python
RecursiveChunking(chunk_size=500, overlap=50)
```

#### SemanticChunking
Groups by semantic similarity.
```python
SemanticChunking(
    embedding_model=HuggingFaceEmbedding(),
    similarity_threshold=0.7,
    max_chunk_size=1000
)
```

## Examples

### Example 1: Document Q&A System

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# Load documents
docs = load_company_documentation()
for doc in docs:
    rag.add_document(doc['content'], doc['metadata'])

rag.process_documents()

# Interactive Q&A
while True:
    question = input("Ask a question: ")
    result = rag.query(question)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Sources: {result['num_sources']} documents")
```

### Example 2: Research Paper Analysis

```python
from rag_system import RAGSystem, SemanticChunking, OpenAIEmbedding

# Configure for academic papers
rag = RAGSystem(
    embedding_model=OpenAIEmbedding(model="text-embedding-3-large"),
    chunking_strategy=SemanticChunking(
        embedding_model=HuggingFaceEmbedding(),
        similarity_threshold=0.8
    ),
    hybrid_search_alpha=0.9  # Emphasize semantic understanding
)

# Add research papers
papers = load_research_papers()
for paper in papers:
    rag.add_document(
        paper['text'],
        metadata={
            'title': paper['title'],
            'authors': paper['authors'],
            'year': paper['year'],
            'venue': paper['venue']
        }
    )

rag.process_documents()

# Analyze research questions
query = "What are the latest advances in transformer architectures?"
result = rag.query(query, top_k=10)

print(result['answer'])
for citation in result['citations']:
    print(f"  - {citation['metadata']['title']} ({citation['metadata']['year']})")
```

### Example 3: Customer Support Knowledge Base

```python
from rag_system import RAGSystem, SentenceChunking

# Setup for customer support
rag = RAGSystem(
    chunking_strategy=SentenceChunking(chunk_size=300),
    enable_query_expansion=True,  # Handle varied question phrasings
    hybrid_search_alpha=0.6  # Balance semantic and keyword matching
)

# Load KB articles
kb_articles = load_knowledge_base()
for article in kb_articles:
    rag.add_document(article['content'], {
        'category': article['category'],
        'product': article['product'],
        'last_updated': article['updated_at']
    })

rag.process_documents()

# Handle customer queries
def handle_support_query(customer_question):
    result = rag.query(customer_question, top_k=3)

    response = {
        'answer': result['answer'],
        'confidence': result['search_results'],
        'related_articles': [
            c['metadata'] for c in result['citations']
        ]
    }

    return response
```

## Performance Considerations

### Embedding Generation
- Use batch processing for large document sets
- Cache embeddings to avoid recomputation
- Consider dimensionality vs. accuracy trade-offs

### Search Optimization
- Hybrid search is slower but more accurate than single-mode
- Adjust `hybrid_search_alpha` based on your use case
- Reranking improves quality but adds latency

### Scaling
- For large datasets (>10k documents), consider vector databases (Pinecone, Weaviate, Qdrant)
- Implement distributed processing for massive document collections
- Use approximate nearest neighbor search for >1M vectors

## Evaluation

```python
# Prepare test set
test_queries = [
    ("What is machine learning?", "expected answer..."),
    ("Explain neural networks", "expected answer..."),
]

# Evaluate
metrics = rag.evaluate(test_queries)

print(f"Retrieval Success Rate: {metrics['retrieval_success_rate']:.2%}")
print(f"Avg Results per Query: {metrics['avg_results_per_query']:.2f}")
print(f"Avg Confidence: {metrics['avg_confidence_score']:.3f}")
```

## Integration with LLMs

### OpenAI Integration

```python
import openai

# In production, replace _generate_llm_response with:
def _generate_llm_response(self, query, context, search_results):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Answer based on context. Cite sources."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content
```

### Anthropic Claude Integration

```python
import anthropic

def _generate_llm_response(self, query, context, search_results):
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a detailed answer based on the context."
            }
        ]
    )
    return message.content[0].text
```

## Troubleshooting

### Common Issues

**Issue:** Slow embedding generation
- **Solution:** Increase batch_size in process_documents()
- **Solution:** Use GPU-accelerated embedding models

**Issue:** Poor search results
- **Solution:** Adjust chunking strategy (smaller/larger chunks)
- **Solution:** Try different embedding models
- **Solution:** Tune hybrid_search_alpha parameter

**Issue:** Irrelevant answers
- **Solution:** Increase top_k to retrieve more context
- **Solution:** Enable reranking
- **Solution:** Use semantic chunking for better coherence

**Issue:** Memory errors with large datasets
- **Solution:** Process documents in batches
- **Solution:** Use lower-dimensional embeddings
- **Solution:** Migrate to vector database backend

## Best Practices

1. **Chunking**: Choose strategy based on document structure
   - Technical docs: SentenceChunking
   - Long-form content: SemanticChunking
   - Mixed content: RecursiveChunking

2. **Embedding Models**: Match to use case
   - English-only: HuggingFace MiniLM (fast, good quality)
   - Multilingual: Cohere or OpenAI
   - Highest quality: OpenAI text-embedding-3-large

3. **Search Configuration**:
   - Technical queries: Higher alpha (0.8-0.9)
   - Varied terminology: Lower alpha (0.5-0.6)
   - Always enable reranking for production

4. **Context Length**: Balance comprehensiveness vs. focus
   - Smaller top_k (3-5) for focused answers
   - Larger top_k (10-15) for comprehensive coverage

5. **Metadata**: Include rich metadata for better attribution
   - Source document
   - Author/date
   - Section/chapter
   - Tags/categories

## Contributing

This is a reference implementation. For production use:
1. Replace demo embeddings with actual API calls
2. Integrate real LLM providers
3. Add authentication and rate limiting
4. Implement comprehensive error handling
5. Add monitoring and logging
6. Consider vector database backends

## License

Copyright 2024 Brill Consulting. All rights reserved.

## Support

For questions or issues, contact Brill Consulting.

## Changelog

### Version 1.0.0 (2024)
- Initial production-ready release
- Multiple embedding model support
- Advanced chunking strategies
- Hybrid search with BM25
- Cross-encoder reranking
- Query expansion
- Citation tracking
- Streaming responses
- Comprehensive documentation

## References

- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval (Karpukhin et al., 2020)](https://arxiv.org/abs/2004.04906)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
