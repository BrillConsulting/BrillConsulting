# RAG System (Retrieval-Augmented Generation)

Document Q&A system combining semantic search with LLM generation for accurate, source-grounded responses.

## Features

- **Document Ingestion**: Process multiple document formats
- **Smart Chunking**: Overlapping chunks for context preservation
- **Vector Embeddings**: OpenAI, HuggingFace, Cohere support
- **Semantic Search**: Find relevant context using cosine similarity
- **Context-Aware Generation**: LLM answers grounded in retrieved docs
- **Source Citation**: Track and cite source documents
- **Index Persistence**: Save/load document indexes

## Technologies

- OpenAI Embeddings / Sentence Transformers
- FAISS / Chroma for vector storage
- NumPy for similarity calculations

## Usage

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem(embedding_model="text-embedding-ada-002")

# Add documents
rag.add_document("Your document text here", metadata={"title": "Doc1"})

# Process and index
rag.process_documents(chunk_size=500)
rag.generate_embeddings()

# Query
result = rag.query("What is machine learning?", top_k=3)
print(result["answer"])
print(result["sources"])
```

## Demo

```bash
python rag_system.py
```

**Note:** For production, integrate actual embedding models and LLM APIs.
