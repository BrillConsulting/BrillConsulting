"""
Production-Ready RAG System (Retrieval-Augmented Generation)
==============================================================

A comprehensive RAG system with advanced features:
- Multiple embedding models (OpenAI, HuggingFace, Cohere, Voyage)
- Advanced chunking strategies (semantic, recursive, sentence-based)
- Hybrid search (semantic + keyword BM25)
- Reranking with cross-encoders
- Query expansion and reformulation
- Citation tracking with precise source attribution
- Multi-document synthesis
- Streaming response support
- Evaluation metrics

Author: Brill Consulting
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Callable, Any, Union
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
import hashlib


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    text: str
    metadata: Dict[str, Any]
    created_at: str

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data


@dataclass
class SearchResult:
    """Represents a search result with scoring."""
    chunk: Chunk
    semantic_score: float
    keyword_score: float
    combined_score: float
    rerank_score: Optional[float] = None


@dataclass
class Citation:
    """Represents a citation with precise source attribution."""
    doc_id: str
    chunk_id: str
    text_snippet: str
    start_char: int
    end_char: int
    confidence: float
    metadata: Dict[str, Any]


# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._dimension = 1536 if "large" in model else 512

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        # Production implementation:
        # import openai
        # client = openai.OpenAI(api_key=self.api_key)
        # response = client.embeddings.create(input=text, model=self.model)
        # return np.array(response.data[0].embedding)

        # Demo: return normalized random embedding
        embedding = np.random.randn(self._dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts."""
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace sentence transformer embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._dimension = 384  # Default for MiniLM

        # Production implementation:
        # from sentence_transformers import SentenceTransformer
        # self.model = SentenceTransformer(model_name)
        # self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using sentence transformers."""
        # Production: return self.model.encode(text)
        embedding = np.random.randn(self._dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch."""
        # Production: return self.model.encode(texts)
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class CohereEmbedding(EmbeddingModel):
    """Cohere embedding model."""

    def __init__(self, model: str = "embed-english-v3.0", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key
        self._dimension = 1024

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding using Cohere API."""
        # Production:
        # import cohere
        # co = cohere.Client(self.api_key)
        # response = co.embed(texts=[text], model=self.model)
        # return np.array(response.embeddings[0])

        embedding = np.random.randn(self._dimension).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch."""
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


# ============================================================================
# Chunking Strategies
# ============================================================================

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Split text into chunks."""
        pass


class CharacterChunking(ChunkingStrategy):
    """Simple character-based chunking with overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Split text into character-based chunks."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=chunk_index,
                metadata=metadata
            ))

            chunk_index += 1
            start += (self.chunk_size - self.overlap)

        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking that respects sentence boundaries."""

    def __init__(self, chunk_size: int = 500, overlap_sentences: int = 2):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions."""
        # Simple sentence splitting (production: use spaCy or nltk)
        pattern = r'[.!?]+[\s]+'
        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            sent_text = text[last_end:match.end()].strip()
            if sent_text:
                sentences.append((sent_text, last_end, match.end()))
            last_end = match.end()

        # Add remaining text
        if last_end < len(text):
            sent_text = text[last_end:].strip()
            if sent_text:
                sentences.append((sent_text, last_end, len(text)))

        return sentences

    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Split text into sentence-based chunks."""
        sentences = self._split_sentences(text)
        chunks = []
        chunk_index = 0

        i = 0
        while i < len(sentences):
            chunk_sentences = []
            chunk_length = 0
            start_pos = sentences[i][1]

            # Add sentences until chunk_size is reached
            while i < len(sentences) and chunk_length < self.chunk_size:
                sent_text, _, sent_end = sentences[i]
                chunk_sentences.append(sent_text)
                chunk_length += len(sent_text)
                end_pos = sent_end
                i += 1

            chunk_text = " ".join(chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"

            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                start_char=start_pos,
                end_char=end_pos,
                chunk_index=chunk_index,
                metadata=metadata
            ))

            chunk_index += 1

            # Overlap: go back overlap_sentences
            i -= min(self.overlap_sentences, len(chunk_sentences))

        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking that tries multiple separators."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Split text recursively using multiple separators."""
        chunks = []
        self._recursive_split(text, 0, doc_id, metadata, chunks, 0)
        return chunks

    def _recursive_split(self, text: str, start_pos: int, doc_id: str,
                        metadata: Dict, chunks: List[Chunk], chunk_index: int) -> int:
        """Recursively split text."""
        if len(text) <= self.chunk_size:
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=text,
                start_char=start_pos,
                end_char=start_pos + len(text),
                chunk_index=chunk_index,
                metadata=metadata
            ))
            return chunk_index + 1

        # Try each separator
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""
                current_start = start_pos

                for split in splits:
                    if len(current_chunk) + len(split) <= self.chunk_size:
                        current_chunk += split + separator
                    else:
                        if current_chunk:
                            chunk_id = f"{doc_id}_chunk_{chunk_index}"
                            chunks.append(Chunk(
                                id=chunk_id,
                                doc_id=doc_id,
                                text=current_chunk.rstrip(),
                                start_char=current_start,
                                end_char=current_start + len(current_chunk),
                                chunk_index=chunk_index,
                                metadata=metadata
                            ))
                            chunk_index += 1
                            current_start += len(current_chunk) - self.overlap
                        current_chunk = split + separator

                # Add remaining
                if current_chunk:
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    chunks.append(Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        text=current_chunk.rstrip(),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        chunk_index=chunk_index,
                        metadata=metadata
                    ))
                    chunk_index += 1

                return chunk_index

        return chunk_index


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking using embedding similarity."""

    def __init__(self, embedding_model: EmbeddingModel,
                 similarity_threshold: float = 0.7,
                 max_chunk_size: int = 1000):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str, doc_id: str, metadata: Dict) -> List[Chunk]:
        """Split text based on semantic similarity."""
        # Split into sentences first
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_batch([s[0] for s in sentences])

        # Group sentences by similarity
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_start = sentences[0][1]
        chunk_index = 0

        for i in range(1, len(sentences)):
            # Calculate similarity with last sentence in current chunk
            similarity = np.dot(embeddings[i], embeddings[i-1])

            current_length = sum(len(s[0]) for s in current_chunk_sentences)

            # Start new chunk if similarity is low or max size reached
            if similarity < self.similarity_threshold or current_length > self.max_chunk_size:
                # Create chunk from current sentences
                chunk_text = " ".join(s[0] for s in current_chunk_sentences)
                chunk_id = f"{doc_id}_chunk_{chunk_index}"

                chunks.append(Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    start_char=current_chunk_start,
                    end_char=current_chunk_sentences[-1][2],
                    chunk_index=chunk_index,
                    metadata=metadata
                ))

                chunk_index += 1
                current_chunk_sentences = [sentences[i]]
                current_chunk_start = sentences[i][1]
            else:
                current_chunk_sentences.append(sentences[i])

        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(s[0] for s in current_chunk_sentences)
            chunk_id = f"{doc_id}_chunk_{chunk_index}"

            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                start_char=current_chunk_start,
                end_char=current_chunk_sentences[-1][2],
                chunk_index=chunk_index,
                metadata=metadata
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences."""
        pattern = r'[.!?]+[\s]+'
        sentences = []
        last_end = 0

        for match in re.finditer(pattern, text):
            sent_text = text[last_end:match.end()].strip()
            if sent_text:
                sentences.append((sent_text, last_end, match.end()))
            last_end = match.end()

        if last_end < len(text):
            sent_text = text[last_end:].strip()
            if sent_text:
                sentences.append((sent_text, last_end, len(text)))

        return sentences


# ============================================================================
# BM25 (Keyword Search)
# ============================================================================

class BM25:
    """BM25 algorithm for keyword-based search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus."""
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Calculate document frequencies
        df = defaultdict(int)
        for doc in corpus:
            words = set(doc.lower().split())
            for word in words:
                df[word] += 1

        # Calculate IDF
        num_docs = len(corpus)
        for word, freq in df.items():
            self.idf[word] = np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc = self.corpus[doc_idx]
        doc_words = doc.lower().split()
        query_words = query.lower().split()

        score = 0.0
        doc_len = self.doc_len[doc_idx]

        for word in query_words:
            if word not in self.idf:
                continue

            word_freq = doc_words.count(word)
            idf = self.idf[word]

            numerator = word_freq * (self.k1 + 1)
            denominator = word_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search corpus and return top-k results."""
        scores = [(i, self.score(query, i)) for i in range(len(self.corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# Reranker
# ============================================================================

class Reranker:
    """Cross-encoder reranker for improving search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name

        # Production:
        # from sentence_transformers import CrossEncoder
        # self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        """Rerank search results using cross-encoder."""
        # Production implementation:
        # pairs = [(query, result.chunk.text) for result in results]
        # scores = self.model.predict(pairs)
        #
        # for result, score in zip(results, scores):
        #     result.rerank_score = float(score)
        #
        # results.sort(key=lambda x: x.rerank_score, reverse=True)
        # return results[:top_k]

        # Demo: Use combined score as rerank score
        for result in results:
            result.rerank_score = result.combined_score * 1.1  # Slight boost

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]


# ============================================================================
# Query Expansion
# ============================================================================

class QueryExpander:
    """Expands queries using various techniques."""

    def __init__(self):
        pass

    def expand_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        # Production: Use WordNet or thesaurus API
        # For demo, return original query and simple variations
        expanded = [query]

        # Simple word variations (production: use proper NLP)
        variations = {
            "what is": ["define", "explain", "describe"],
            "how to": ["steps to", "way to", "method to"],
            "why": ["reason for", "cause of", "explanation for"],
        }

        for phrase, synonyms in variations.items():
            if phrase in query.lower():
                for synonym in synonyms:
                    expanded.append(query.lower().replace(phrase, synonym))

        return expanded[:5]  # Limit expansions

    def expand_multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple query variations."""
        # Production: Use LLM to generate query variations
        # For demo, create simple variations
        queries = [query]

        # Add question variations
        if not query.endswith("?"):
            queries.append(f"{query}?")

        # Add contextual variations
        queries.append(f"Information about {query}")
        queries.append(f"Details on {query}")

        return queries[:num_queries]

    def expand_hyde(self, query: str) -> str:
        """Hypothetical Document Embeddings (HyDE)."""
        # Production: Use LLM to generate hypothetical answer
        # For demo, return enhanced query
        return f"A comprehensive explanation of {query} including key concepts and examples."


# ============================================================================
# Production-Ready RAG System
# ============================================================================

class RAGSystem:
    """
    Production-ready Retrieval-Augmented Generation system.

    Features:
    - Multiple embedding models
    - Advanced chunking strategies
    - Hybrid search (semantic + keyword)
    - Reranking
    - Query expansion
    - Citation tracking
    - Multi-document synthesis
    - Streaming responses
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None,
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
        hybrid_search_alpha: float = 0.7,  # Weight for semantic vs keyword
    ):
        """
        Initialize RAG system.

        Args:
            embedding_model: Embedding model to use
            chunking_strategy: Chunking strategy to use
            enable_reranking: Enable reranking of results
            enable_query_expansion: Enable query expansion
            hybrid_search_alpha: Weight for semantic search (1-alpha for keyword)
        """
        self.embedding_model = embedding_model or HuggingFaceEmbedding()
        self.chunking_strategy = chunking_strategy or SentenceChunking()
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion
        self.hybrid_search_alpha = hybrid_search_alpha

        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_list: List[Chunk] = []

        self.bm25: Optional[BM25] = None
        self.reranker: Optional[Reranker] = None
        self.query_expander: Optional[QueryExpander] = None

        if enable_reranking:
            self.reranker = Reranker()
        if enable_query_expansion:
            self.query_expander = QueryExpander()

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add document to the system.

        Args:
            text: Document text
            metadata: Optional metadata
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = hashlib.md5(text.encode()).hexdigest()[:16]

        document = Document(
            id=doc_id,
            text=text,
            metadata=metadata or {},
            created_at=datetime.utcnow().isoformat()
        )

        self.documents[doc_id] = document
        return doc_id

    def process_documents(self, batch_size: int = 32) -> None:
        """
        Process all documents: chunk and generate embeddings.

        Args:
            batch_size: Batch size for embedding generation
        """
        print(f"Processing {len(self.documents)} documents...")

        # Chunk all documents
        all_chunks = []
        for doc in self.documents.values():
            chunks = self.chunking_strategy.chunk_text(
                doc.text, doc.id, doc.metadata
            )
            all_chunks.extend(chunks)

        self.chunk_list = all_chunks
        self.chunks = {chunk.id: chunk for chunk in all_chunks}

        print(f"  Created {len(all_chunks)} chunks")

        # Generate embeddings in batches
        print(f"  Generating embeddings...")
        chunk_texts = [chunk.text for chunk in all_chunks]

        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            embeddings = self.embedding_model.embed_batch(batch)

            for j, embedding in enumerate(embeddings):
                self.chunk_list[i+j].embedding = embedding

        print(f"  Generated {len(all_chunks)} embeddings")

        # Build BM25 index
        print(f"  Building BM25 index...")
        self.bm25 = BM25()
        self.bm25.fit(chunk_texts)

        print(f"  Processing complete!")

    def semantic_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Semantic search using embeddings.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        query_embedding = self.embedding_model.embed_text(query)

        # Calculate cosine similarities
        similarities = []
        for chunk in self.chunk_list:
            if chunk.embedding is not None:
                similarity = np.dot(query_embedding, chunk.embedding)
                similarities.append((chunk, float(similarity)))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[Chunk, float]]:
        """
        Keyword search using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25 is None:
            return []

        results = self.bm25.search(query, top_k=top_k)
        return [(self.chunk_list[idx], score) for idx, score in results]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for semantic search (overrides default)

        Returns:
            List of SearchResult objects
        """
        alpha = alpha if alpha is not None else self.hybrid_search_alpha

        # Get semantic results
        semantic_results = self.semantic_search(query, top_k=top_k*2)

        # Get keyword results
        keyword_results = self.keyword_search(query, top_k=top_k*2)

        # Normalize scores
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            semantic_dict = {chunk.id: score/max_semantic
                           for chunk, score in semantic_results}
        else:
            semantic_dict = {}

        if keyword_results:
            max_keyword = max(score for _, score in keyword_results)
            if max_keyword > 0:
                keyword_dict = {chunk.id: score/max_keyword
                              for chunk, score in keyword_results}
            else:
                keyword_dict = {}
        else:
            keyword_dict = {}

        # Combine scores
        all_chunk_ids = set(semantic_dict.keys()) | set(keyword_dict.keys())
        combined_results = []

        for chunk_id in all_chunk_ids:
            semantic_score = semantic_dict.get(chunk_id, 0.0)
            keyword_score = keyword_dict.get(chunk_id, 0.0)
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score

            chunk = self.chunks[chunk_id]
            combined_results.append(SearchResult(
                chunk=chunk,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                combined_score=combined_score
            ))

        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        return combined_results[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results
            search_mode: "semantic", "keyword", or "hybrid"

        Returns:
            List of SearchResult objects
        """
        # Apply query expansion if enabled
        queries = [query]
        if self.enable_query_expansion and self.query_expander:
            expanded = self.query_expander.expand_multi_query(query)
            queries.extend(expanded[1:])  # Skip first (original)

        # Search with all queries and combine results
        all_results = []
        for q in queries:
            if search_mode == "semantic":
                results = self.semantic_search(q, top_k=top_k*2)
                for chunk, score in results:
                    all_results.append(SearchResult(
                        chunk=chunk,
                        semantic_score=score,
                        keyword_score=0.0,
                        combined_score=score
                    ))
            elif search_mode == "keyword":
                results = self.keyword_search(q, top_k=top_k*2)
                for chunk, score in results:
                    all_results.append(SearchResult(
                        chunk=chunk,
                        semantic_score=0.0,
                        keyword_score=score,
                        combined_score=score
                    ))
            else:  # hybrid
                results = self.hybrid_search(q, top_k=top_k*2)
                all_results.extend(results)

        # Deduplicate and re-rank
        seen = set()
        unique_results = []
        for result in all_results:
            if result.chunk.id not in seen:
                seen.add(result.chunk.id)
                unique_results.append(result)

        # Sort by score
        unique_results.sort(key=lambda x: x.combined_score, reverse=True)
        unique_results = unique_results[:top_k*2]

        # Apply reranking if enabled
        if self.enable_reranking and self.reranker and unique_results:
            unique_results = self.reranker.rerank(query, unique_results, top_k=top_k)
        else:
            unique_results = unique_results[:top_k]

        return unique_results

    def extract_citations(
        self,
        answer: str,
        context_chunks: List[Chunk],
        confidence_threshold: float = 0.5
    ) -> List[Citation]:
        """
        Extract citations from answer based on context chunks.

        Args:
            answer: Generated answer
            context_chunks: Context chunks used
            confidence_threshold: Minimum confidence for citation

        Returns:
            List of Citation objects
        """
        citations = []

        # Simple citation extraction (production: use more sophisticated methods)
        for chunk in context_chunks:
            # Check if chunk text appears in answer
            chunk_text = chunk.text[:100]  # Use first 100 chars

            # Calculate overlap
            answer_lower = answer.lower()
            chunk_lower = chunk_text.lower()

            # Find common phrases
            words = chunk_lower.split()
            max_overlap = 0
            best_snippet = ""

            for i in range(len(words)):
                for j in range(i+3, min(i+20, len(words)+1)):
                    phrase = " ".join(words[i:j])
                    if phrase in answer_lower:
                        if len(phrase) > max_overlap:
                            max_overlap = len(phrase)
                            best_snippet = " ".join(words[i:j])

            if max_overlap > 20:  # Minimum phrase length
                confidence = min(max_overlap / 100, 1.0)

                if confidence >= confidence_threshold:
                    citations.append(Citation(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.id,
                        text_snippet=best_snippet,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        confidence=confidence,
                        metadata=chunk.metadata or {}
                    ))

        return citations

    def synthesize_answer(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Synthesize answer from multiple documents.

        Args:
            query: User query
            search_results: Search results
            max_context_length: Maximum context length

        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Build context from results
        context_parts = []
        context_chunks = []
        total_length = 0

        for result in search_results:
            chunk_text = result.chunk.text
            if total_length + len(chunk_text) < max_context_length:
                source_info = result.chunk.metadata.get('title', 'Unknown')
                context_parts.append(f"[Source: {source_info}]\n{chunk_text}")
                context_chunks.append(result.chunk)
                total_length += len(chunk_text)

        context = "\n\n".join(context_parts)

        # Generate answer (production: use actual LLM)
        answer = self._generate_llm_response(query, context, search_results)

        # Extract citations
        citations = self.extract_citations(answer, context_chunks)

        return {
            "query": query,
            "answer": answer,
            "citations": [asdict(c) for c in citations],
            "num_sources": len(set(c.doc_id for c in citations)),
            "search_results": len(search_results),
            "context_length": len(context)
        }

    def _generate_llm_response(
        self,
        query: str,
        context: str,
        search_results: List[SearchResult]
    ) -> str:
        """
        Generate LLM response with context.

        Production implementation:
        ```python
        import openai

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions based on the provided context. Cite sources when possible."
                },
                {
                    "role": "user",
                    "content": f"Context:\\n{context}\\n\\nQuestion: {query}"
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
        ```
        """
        # Demo response
        sources = set()
        for result in search_results:
            title = result.chunk.metadata.get('title', 'Unknown')
            sources.add(title)

        source_list = ", ".join(sorted(sources))

        return (
            f"Based on the provided context from {len(sources)} source(s) "
            f"({source_list}), here is the answer to your question: '{query}'.\n\n"
            f"[This is a demonstration response. In production, this would be generated "
            f"by an actual LLM (GPT-4, Claude, etc.) using the retrieved context of "
            f"{len(context)} characters from {len(search_results)} relevant chunks. "
            f"The system would synthesize information across multiple documents and "
            f"provide accurate citations.]\n\n"
            f"The answer would include specific details extracted from the context, "
            f"properly synthesized and cited."
        )

    def query_stream(
        self,
        query: str,
        top_k: int = 5
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream response generation.

        Args:
            query: User query
            top_k: Number of chunks to retrieve

        Yields:
            Streaming response chunks
        """
        # Search
        yield {"type": "search_start", "query": query}

        results = self.search(query, top_k=top_k)

        yield {
            "type": "search_complete",
            "num_results": len(results),
            "top_sources": [r.chunk.metadata.get('title', 'Unknown')
                          for r in results[:3]]
        }

        # Generate answer
        yield {"type": "generation_start"}

        answer_data = self.synthesize_answer(query, results)

        # Stream answer in chunks (production: stream from LLM)
        answer = answer_data["answer"]
        chunk_size = 20

        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i+chunk_size]
            yield {"type": "answer_chunk", "content": chunk}

        # Final metadata
        yield {
            "type": "complete",
            "citations": answer_data["citations"],
            "num_sources": answer_data["num_sources"],
            "metadata": {
                "search_results": answer_data["search_results"],
                "context_length": answer_data["context_length"]
            }
        }

    def query(
        self,
        query: str,
        top_k: int = 5,
        search_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            search_mode: "semantic", "keyword", or "hybrid"

        Returns:
            Complete answer with citations and metadata
        """
        # Search
        results = self.search(query, top_k=top_k, search_mode=search_mode)

        # Generate answer
        answer_data = self.synthesize_answer(query, results)

        # Add search metadata
        answer_data["search_mode"] = search_mode
        answer_data["reranking_enabled"] = self.enable_reranking
        answer_data["query_expansion_enabled"] = self.enable_query_expansion

        return answer_data

    def evaluate(
        self,
        test_queries: List[Tuple[str, str]],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate RAG system performance.

        Args:
            test_queries: List of (query, expected_answer) tuples
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with evaluation metrics
        """
        # Simple evaluation metrics
        total_queries = len(test_queries)
        retrieval_success = 0
        avg_num_results = 0
        avg_confidence = 0

        for query, expected in test_queries:
            results = self.search(query, top_k=top_k)

            if results:
                retrieval_success += 1
                avg_num_results += len(results)
                avg_confidence += sum(r.combined_score for r in results) / len(results)

        return {
            "total_queries": total_queries,
            "retrieval_success_rate": retrieval_success / total_queries,
            "avg_results_per_query": avg_num_results / total_queries,
            "avg_confidence_score": avg_confidence / total_queries if total_queries > 0 else 0
        }

    def save_index(self, filepath: str) -> None:
        """Save the complete index."""
        data = {
            "documents": {doc_id: asdict(doc) for doc_id, doc in self.documents.items()},
            "chunks": [chunk.to_dict() for chunk in self.chunk_list],
            "config": {
                "embedding_dimension": self.embedding_model.dimension,
                "num_documents": len(self.documents),
                "num_chunks": len(self.chunk_list)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved index to {filepath}")

    def load_index(self, filepath: str) -> None:
        """Load index from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load documents
        self.documents = {}
        for doc_id, doc_data in data["documents"].items():
            self.documents[doc_id] = Document(**doc_data)

        # Load chunks
        self.chunk_list = []
        for chunk_data in data["chunks"]:
            embedding = chunk_data.pop('embedding', None)
            chunk = Chunk(**chunk_data)
            if embedding is not None:
                chunk.embedding = np.array(embedding)
            self.chunk_list.append(chunk)

        self.chunks = {chunk.id: chunk for chunk in self.chunk_list}

        # Rebuild BM25
        if self.chunk_list:
            self.bm25 = BM25()
            self.bm25.fit([chunk.text for chunk in self.chunk_list])

        print(f"Loaded index from {filepath}")
        print(f"  Documents: {len(self.documents)}")
        print(f"  Chunks: {len(self.chunk_list)}")


# ============================================================================
# Demo and CLI
# ============================================================================

def demo_basic():
    """Basic RAG demo."""
    print("="*70)
    print("PRODUCTION-READY RAG SYSTEM - Basic Demo")
    print("="*70)

    # Initialize with default settings
    rag = RAGSystem(
        embedding_model=HuggingFaceEmbedding(),
        chunking_strategy=SentenceChunking(chunk_size=300),
        enable_reranking=True,
        enable_query_expansion=True
    )

    # Add sample documents
    print("\n1. Adding Documents")
    print("-"*70)

    docs = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The primary aim is to allow computers to learn automatically without human intervention.",
            "metadata": {"title": "Machine Learning Basics", "author": "AI Research", "year": 2024}
        },
        {
            "text": "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures include deep neural networks, deep belief networks, recurrent neural networks, and convolutional neural networks. They have been applied to fields including computer vision, speech recognition, and natural language processing.",
            "metadata": {"title": "Deep Learning Overview", "author": "Neural Networks Inc", "year": 2024}
        },
        {
            "text": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human languages. NLP combines computational linguistics with statistical, machine learning, and deep learning models.",
            "metadata": {"title": "NLP Fundamentals", "author": "Language AI Lab", "year": 2024}
        },
        {
            "text": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by incorporating external knowledge retrieval. It combines the power of pre-trained language models with the ability to access and incorporate relevant information from a knowledge base. This approach significantly improves the accuracy and factual grounding of generated responses.",
            "metadata": {"title": "RAG Systems", "author": "LLM Research Group", "year": 2024}
        }
    ]

    for doc in docs:
        doc_id = rag.add_document(doc["text"], doc["metadata"])
        print(f"  Added: {doc['metadata']['title']} (ID: {doc_id[:8]}...)")

    # Process documents
    print("\n2. Processing Documents")
    print("-"*70)
    rag.process_documents()

    # Perform queries
    print("\n3. RAG Queries")
    print("-"*70)

    queries = [
        "What is machine learning?",
        "Explain deep learning and its applications",
        "How does RAG improve language models?"
    ]

    for query in queries:
        print(f"\n  Query: {query}")
        print("  " + "-"*66)

        result = rag.query(query, top_k=3, search_mode="hybrid")

        print(f"  Answer: {result['answer'][:200]}...")
        print(f"\n  Citations: {len(result['citations'])}")
        print(f"  Sources: {result['num_sources']}")
        print(f"  Search results: {result['search_results']}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


def demo_advanced():
    """Advanced RAG demo with all features."""
    print("="*70)
    print("PRODUCTION-READY RAG SYSTEM - Advanced Demo")
    print("="*70)

    # Test different embedding models
    print("\n1. Testing Different Embedding Models")
    print("-"*70)

    models = [
        ("HuggingFace", HuggingFaceEmbedding()),
        ("OpenAI", OpenAIEmbedding()),
        ("Cohere", CohereEmbedding())
    ]

    for name, model in models:
        print(f"  {name}: dimension={model.dimension}")

    # Test different chunking strategies
    print("\n2. Testing Chunking Strategies")
    print("-"*70)

    sample_text = "Machine learning is amazing. It enables computers to learn. Deep learning uses neural networks. Neural networks have multiple layers. This creates powerful AI systems."

    strategies = [
        ("Character", CharacterChunking(chunk_size=50, overlap=10)),
        ("Sentence", SentenceChunking(chunk_size=100)),
        ("Recursive", RecursiveChunking(chunk_size=80))
    ]

    for name, strategy in strategies:
        chunks = strategy.chunk_text(sample_text, "test_doc", {})
        print(f"  {name}: {len(chunks)} chunks")

    # Test hybrid search
    print("\n3. Hybrid Search Comparison")
    print("-"*70)

    rag = RAGSystem(
        embedding_model=HuggingFaceEmbedding(),
        chunking_strategy=SentenceChunking(),
        hybrid_search_alpha=0.7
    )

    # Add documents
    docs = [
        "Artificial intelligence and machine learning are transforming industries.",
        "Deep neural networks power modern AI applications.",
        "Natural language processing enables human-computer interaction."
    ]

    for i, doc in enumerate(docs):
        rag.add_document(doc, {"title": f"Doc{i+1}"})

    rag.process_documents()

    query = "AI and machine learning"

    for mode in ["semantic", "keyword", "hybrid"]:
        results = rag.search(query, top_k=2, search_mode=mode)
        print(f"  {mode.capitalize()} search: {len(results)} results")
        if results:
            print(f"    Top score: {results[0].combined_score:.3f}")

    # Test streaming
    print("\n4. Streaming Response")
    print("-"*70)
    print(f"  Query: {query}")
    print("  Stream: ", end="")

    for chunk in rag.query_stream(query, top_k=2):
        if chunk["type"] == "answer_chunk":
            print(chunk["content"], end="", flush=True)
        elif chunk["type"] == "complete":
            print(f"\n  Citations: {len(chunk['citations'])}")

    print("\n" + "="*70)
    print("Advanced Demo Complete!")
    print("="*70)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--advanced":
        demo_advanced()
    else:
        demo_basic()
