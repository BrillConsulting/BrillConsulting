"""
RAG System (Retrieval-Augmented Generation)
===========================================

Retrieval-Augmented Generation system for document Q&A:
- Document ingestion and chunking
- Vector embeddings (OpenAI, HuggingFace)
- Vector storage (FAISS, Chroma)
- Semantic search
- Context-aware generation
- Source citation

Author: Brill Consulting
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import json


class RAGSystem:
    """Retrieval-Augmented Generation system."""

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize RAG system.

        Args:
            embedding_model: Model for generating embeddings
        """
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []
        self.chunks = []

    def add_document(self, document: str, metadata: Optional[Dict] = None):
        """
        Add document to knowledge base.

        Args:
            document: Document text
            metadata: Optional metadata (title, source, etc.)
        """
        doc_id = len(self.documents)
        self.documents.append({
            "id": doc_id,
            "text": document,
            "metadata": metadata or {}
        })

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)

        return chunks

    def process_documents(self, chunk_size: int = 500, overlap: int = 50):
        """Process and chunk all documents."""
        self.chunks = []

        for doc in self.documents:
            doc_chunks = self.chunk_text(doc["text"], chunk_size, overlap)

            for idx, chunk in enumerate(doc_chunks):
                self.chunks.append({
                    "doc_id": doc["id"],
                    "chunk_id": idx,
                    "text": chunk,
                    "metadata": doc["metadata"]
                })

        print(f"✓ Processed {len(self.documents)} documents into {len(self.chunks)} chunks")

    def generate_embeddings(self):
        """
        Generate embeddings for all chunks.

        In production, use:
        - OpenAI: openai.Embedding.create()
        - HuggingFace: sentence_transformers
        - Cohere: cohere.embed()
        """
        # Placeholder: random embeddings
        # In production, replace with actual embeddings
        embedding_dim = 384  # Common dimension for sentence transformers

        self.embeddings = []
        for chunk in self.chunks:
            # Simulate embedding generation
            # In production: embedding = model.encode(chunk["text"])
            embedding = np.random.randn(embedding_dim).astype(np.float32)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            self.embeddings.append(embedding)

        print(f"✓ Generated {len(self.embeddings)} embeddings")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant chunks with scores
        """
        if not self.embeddings:
            print("No embeddings available. Run generate_embeddings() first.")
            return []

        # Generate query embedding
        # In production: query_embedding = model.encode(query)
        query_embedding = np.random.randn(len(self.embeddings[0])).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Calculate cosine similarity
        similarities = []
        for idx, emb in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, emb)
            similarities.append((idx, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top_k results
        results = []
        for idx, score in similarities[:top_k]:
            result = self.chunks[idx].copy()
            result["similarity_score"] = float(score)
            results.append(result)

        return results

    def generate_answer(self, query: str, context_chunks: List[Dict],
                       max_context_length: int = 2000) -> Dict:
        """
        Generate answer using retrieved context.

        Args:
            query: User question
            context_chunks: Retrieved relevant chunks
            max_context_length: Maximum context length

        Returns:
            Answer with sources
        """
        # Build context from chunks
        context = ""
        sources = []

        for chunk in context_chunks:
            if len(context) + len(chunk["text"]) < max_context_length:
                context += chunk["text"] + "\n\n"
                sources.append({
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "metadata": chunk["metadata"]
                })

        # Generate answer
        # In production, call LLM with context
        answer = self._generate_llm_response(query, context)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "context_used": context[:200] + "..."  # Preview
        }

    def _generate_llm_response(self, query: str, context: str) -> str:
        """
        Generate LLM response with context.

        In production, implement:
        ```
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
        ```
        """
        # Placeholder response
        return f"Based on the provided context, here's the answer to '{query}'. [This is a demo response. In production, this would use actual LLM with the retrieved context.]"

    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        End-to-end RAG query.

        Args:
            question: User question
            top_k: Number of chunks to retrieve

        Returns:
            Answer with sources and metadata
        """
        # Search for relevant chunks
        relevant_chunks = self.search(question, top_k=top_k)

        # Generate answer
        result = self.generate_answer(question, relevant_chunks)

        return result

    def save_index(self, filepath: str):
        """Save document index and embeddings."""
        data = {
            "documents": self.documents,
            "chunks": self.chunks,
            "embeddings": [emb.tolist() for emb in self.embeddings]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

        print(f"✓ Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load document index and embeddings."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.documents = data["documents"]
        self.chunks = data["chunks"]
        self.embeddings = [np.array(emb) for emb in data["embeddings"]]

        print(f"✓ Loaded index from {filepath}")


def demo():
    """Demo RAG system."""
    print("RAG System Demo")
    print("="*50)

    # Initialize
    rag = RAGSystem(embedding_model="text-embedding-ada-002")

    # Add sample documents
    print("\n1. Adding Documents")
    print("-"*50)

    docs = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning. Common algorithms include decision trees, neural networks, and support vector machines.",
            "metadata": {"title": "ML Basics", "source": "doc1.txt"}
        },
        {
            "text": "Deep learning is a type of machine learning based on artificial neural networks. It uses multiple layers to progressively extract higher-level features. Applications include image recognition, natural language processing, and speech recognition.",
            "metadata": {"title": "Deep Learning", "source": "doc2.txt"}
        },
        {
            "text": "Natural Language Processing (NLP) is a field of AI focused on human-computer language interaction. It includes tasks like text classification, named entity recognition, machine translation, and sentiment analysis. Modern NLP relies heavily on transformer models.",
            "metadata": {"title": "NLP Overview", "source": "doc3.txt"}
        }
    ]

    for doc in docs:
        rag.add_document(doc["text"], doc["metadata"])

    print(f"Added {len(docs)} documents")

    # Process documents
    print("\n2. Processing Documents")
    print("-"*50)
    rag.process_documents(chunk_size=200, overlap=50)

    # Generate embeddings
    print("\n3. Generating Embeddings")
    print("-"*50)
    rag.generate_embeddings()

    # Search
    print("\n4. Semantic Search")
    print("-"*50)
    query = "What is machine learning?"
    results = rag.search(query, top_k=2)

    print(f"Query: {query}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['similarity_score']:.4f}")
        print(f"   Source: {result['metadata']['title']}")
        print(f"   Text: {result['text'][:100]}...")

    # Full RAG query
    print("\n5. RAG Query with Answer Generation")
    print("-"*50)
    questions = [
        "What is deep learning?",
        "What are NLP applications?",
        "What algorithms are used in machine learning?"
    ]

    for question in questions:
        result = rag.query(question, top_k=2)
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['metadata']['title'] for s in result['sources']]}")

    # Save index
    print("\n6. Saving Index")
    print("-"*50)
    rag.save_index("rag_index.json")

    print("\n✓ RAG Demo Complete!")


if __name__ == '__main__':
    demo()
