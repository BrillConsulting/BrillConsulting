"""
VectorDatabases - Production-Ready Vector Database Management System
Author: BrillConsulting
Description: Comprehensive vector database system with multiple backend support
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import numpy as np
import json
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Supported distance metrics for similarity search"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class IndexType(Enum):
    """Types of vector indexes"""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    LSH = "lsh"


@dataclass
class VectorDocument:
    """Document representation with vector embedding"""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'vector': self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            'metadata': self.metadata,
            'text': self.text,
            'timestamp': self.timestamp
        }


@dataclass
class SearchResult:
    """Search result with score and document"""
    id: str
    score: float
    document: VectorDocument
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'score': self.score,
            'rank': self.rank,
            'document': self.document.to_dict()
        }


@dataclass
class IndexConfig:
    """Configuration for vector index"""
    index_type: IndexType = IndexType.FLAT
    dimension: int = 768
    metric: DistanceMetric = DistanceMetric.COSINE
    nlist: int = 100  # For IVF
    m: int = 16  # For HNSW
    ef_construction: int = 200  # For HNSW
    ef_search: int = 50  # For HNSW


class VectorDatabaseBackend(ABC):
    """Abstract base class for vector database backends"""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.documents: Dict[str, VectorDocument] = {}

    @abstractmethod
    def create_index(self) -> bool:
        """Create or initialize the vector index"""
        pass

    @abstractmethod
    def insert(self, document: VectorDocument) -> bool:
        """Insert a single document"""
        pass

    @abstractmethod
    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Insert multiple documents in batch"""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Retrieve document by ID"""
        pass

    @abstractmethod
    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update existing document"""
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID"""
        pass

    @abstractmethod
    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Delete multiple documents"""
        pass

    @abstractmethod
    def optimize_index(self) -> Dict[str, Any]:
        """Optimize index for better performance"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        pass


class FAISSBackend(VectorDatabaseBackend):
    """FAISS vector database backend"""

    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self.index = None
        self.id_map = {}  # Maps string IDs to index positions
        self.create_index()

    def create_index(self) -> bool:
        """Create FAISS index"""
        try:
            import faiss

            if self.config.index_type == IndexType.FLAT:
                if self.config.metric == DistanceMetric.COSINE:
                    self.index = faiss.IndexFlatIP(self.config.dimension)
                else:
                    self.index = faiss.IndexFlatL2(self.config.dimension)

            elif self.config.index_type == IndexType.IVF:
                quantizer = faiss.IndexFlatL2(self.config.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.config.dimension, self.config.nlist
                )

            elif self.config.index_type == IndexType.HNSW:
                self.index = faiss.IndexHNSWFlat(self.config.dimension, self.config.m)
                self.index.hnsw.efConstruction = self.config.ef_construction
                self.index.hnsw.efSearch = self.config.ef_search

            logger.info(f"FAISS index created: {self.config.index_type.value}")
            return True

        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return False

    def insert(self, document: VectorDocument) -> bool:
        """Insert single document into FAISS"""
        try:
            import faiss

            vector = document.vector
            if self.config.metric == DistanceMetric.COSINE:
                # Normalize for cosine similarity
                faiss.normalize_L2(vector.reshape(1, -1))

            # Add to index
            self.index.add(vector.reshape(1, -1))
            position = self.index.ntotal - 1
            self.id_map[document.id] = position
            self.documents[document.id] = document

            logger.debug(f"Document {document.id} inserted at position {position}")
            return True

        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return False

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert documents into FAISS"""
        try:
            import faiss

            vectors = np.vstack([doc.vector for doc in documents])

            if self.config.metric == DistanceMetric.COSINE:
                faiss.normalize_L2(vectors)

            start_pos = self.index.ntotal
            self.index.add(vectors)

            for i, doc in enumerate(documents):
                position = start_pos + i
                self.id_map[doc.id] = position
                self.documents[doc.id] = doc

            return {
                'status': 'success',
                'inserted': len(documents),
                'total': self.index.ntotal
            }

        except Exception as e:
            logger.error(f"Error in batch insert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in FAISS"""
        try:
            import faiss

            query = query_vector.reshape(1, -1)
            if self.config.metric == DistanceMetric.COSINE:
                faiss.normalize_L2(query)

            distances, indices = self.index.search(query, min(top_k * 2, self.index.ntotal))

            results = []
            reverse_map = {v: k for k, v in self.id_map.items()}

            for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx == -1:
                    continue

                doc_id = reverse_map.get(idx)
                if not doc_id:
                    continue

                document = self.documents.get(doc_id)
                if not document:
                    continue

                # Apply filters
                if filters and not self._apply_filters(document, filters):
                    continue

                score = float(dist) if self.config.metric != DistanceMetric.COSINE else float(1 - dist)
                results.append(SearchResult(
                    id=doc_id,
                    score=score,
                    document=document,
                    rank=rank
                ))

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def _apply_filters(self, document: VectorDocument, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters to document"""
        for key, value in filters.items():
            if key not in document.metadata:
                return False
            if document.metadata[key] != value:
                return False
        return True

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document (delete and re-insert)"""
        if doc_id not in self.documents:
            return False
        # Note: FAISS doesn't support direct updates, need to rebuild index
        self.documents[doc_id] = document
        logger.warning("FAISS update requires index rebuild for vector changes")
        return True

    def delete(self, doc_id: str) -> bool:
        """Delete document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            if doc_id in self.id_map:
                del self.id_map[doc_id]
            logger.warning("FAISS delete requires index rebuild to reclaim space")
            return True
        return False

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete documents"""
        deleted = 0
        for doc_id in doc_ids:
            if self.delete(doc_id):
                deleted += 1
        return {'status': 'success', 'deleted': deleted}

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize FAISS index"""
        try:
            if self.config.index_type == IndexType.IVF:
                # Train IVF index
                if not self.index.is_trained and len(self.documents) > self.config.nlist:
                    vectors = np.vstack([doc.vector for doc in self.documents.values()])
                    self.index.train(vectors)
                    return {'status': 'success', 'trained': True}

            return {'status': 'success', 'message': 'Index optimized'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        return {
            'backend': 'FAISS',
            'index_type': self.config.index_type.value,
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.config.dimension,
            'metric': self.config.metric.value,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


class ChromaBackend(VectorDatabaseBackend):
    """ChromaDB vector database backend"""

    def __init__(self, config: IndexConfig, collection_name: str = "default"):
        super().__init__(config)
        self.collection = None
        self.collection_name = collection_name
        self.client = None
        self.create_index()

    def create_index(self) -> bool:
        """Create ChromaDB collection"""
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ))

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"dimension": self.config.dimension}
            )

            logger.info(f"ChromaDB collection created: {self.collection_name}")
            return True

        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Error creating ChromaDB collection: {str(e)}")
            return False

    def insert(self, document: VectorDocument) -> bool:
        """Insert document into ChromaDB"""
        try:
            self.collection.add(
                ids=[document.id],
                embeddings=[document.vector.tolist()],
                metadatas=[document.metadata],
                documents=[document.text] if document.text else None
            )
            self.documents[document.id] = document
            return True

        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return False

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert into ChromaDB"""
        try:
            ids = [doc.id for doc in documents]
            embeddings = [doc.vector.tolist() for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            texts = [doc.text for doc in documents if doc.text]

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts if texts else None
            )

            for doc in documents:
                self.documents[doc.id] = doc

            return {
                'status': 'success',
                'inserted': len(documents),
                'total': self.collection.count()
            }

        except Exception as e:
            logger.error(f"Error in batch insert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in ChromaDB"""
        try:
            where = filters if filters else None

            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where
            )

            search_results = []
            if results['ids'] and len(results['ids']) > 0:
                for rank, (doc_id, distance) in enumerate(zip(results['ids'][0], results['distances'][0])):
                    document = self.documents.get(doc_id)
                    if document:
                        search_results.append(SearchResult(
                            id=doc_id,
                            score=float(1 - distance),  # Convert distance to similarity
                            document=document,
                            rank=rank
                        ))

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document in ChromaDB"""
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[document.vector.tolist()],
                metadatas=[document.metadata],
                documents=[document.text] if document.text else None
            )
            self.documents[doc_id] = document
            return True

        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def delete(self, doc_id: str) -> bool:
        """Delete document from ChromaDB"""
        try:
            self.collection.delete(ids=[doc_id])
            if doc_id in self.documents:
                del self.documents[doc_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete from ChromaDB"""
        try:
            self.collection.delete(ids=doc_ids)
            deleted = 0
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted += 1
            return {'status': 'success', 'deleted': deleted}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize ChromaDB index"""
        return {'status': 'success', 'message': 'ChromaDB handles optimization automatically'}

    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        return {
            'backend': 'ChromaDB',
            'collection_name': self.collection_name,
            'total_vectors': self.collection.count(),
            'dimension': self.config.dimension,
            'metric': self.config.metric.value
        }


class PineconeBackend(VectorDatabaseBackend):
    """Pinecone vector database backend"""

    def __init__(self, config: IndexConfig, api_key: str, environment: str, index_name: str = "default"):
        super().__init__(config)
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        self.create_index()

    def create_index(self) -> bool:
        """Create Pinecone index"""
        try:
            import pinecone

            pinecone.init(api_key=self.api_key, environment=self.environment)

            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric.value
                )

            self.index = pinecone.Index(self.index_name)
            logger.info(f"Pinecone index created/connected: {self.index_name}")
            return True

        except ImportError:
            logger.error("Pinecone not installed. Install with: pip install pinecone-client")
            return False
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {str(e)}")
            return False

    def insert(self, document: VectorDocument) -> bool:
        """Insert document into Pinecone"""
        try:
            self.index.upsert(
                vectors=[(
                    document.id,
                    document.vector.tolist(),
                    document.metadata
                )]
            )
            self.documents[document.id] = document
            return True

        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return False

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert into Pinecone"""
        try:
            vectors = [
                (doc.id, doc.vector.tolist(), doc.metadata)
                for doc in documents
            ]

            # Pinecone has batch size limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)

            for doc in documents:
                self.documents[doc.id] = doc

            return {
                'status': 'success',
                'inserted': len(documents),
                'total': self.index.describe_index_stats()['total_vector_count']
            }

        except Exception as e:
            logger.error(f"Error in batch insert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in Pinecone"""
        try:
            results = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                filter=filters,
                include_metadata=True
            )

            search_results = []
            for rank, match in enumerate(results['matches']):
                doc_id = match['id']
                score = match['score']

                document = self.documents.get(doc_id)
                if document:
                    search_results.append(SearchResult(
                        id=doc_id,
                        score=float(score),
                        document=document,
                        rank=rank
                    ))

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document in Pinecone"""
        return self.insert(document)  # Upsert handles updates

    def delete(self, doc_id: str) -> bool:
        """Delete document from Pinecone"""
        try:
            self.index.delete(ids=[doc_id])
            if doc_id in self.documents:
                del self.documents[doc_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete from Pinecone"""
        try:
            # Delete in batches
            batch_size = 1000
            deleted = 0
            for i in range(0, len(doc_ids), batch_size):
                batch = doc_ids[i:i + batch_size]
                self.index.delete(ids=batch)
                for doc_id in batch:
                    if doc_id in self.documents:
                        del self.documents[doc_id]
                        deleted += 1

            return {'status': 'success', 'deleted': deleted}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize Pinecone index"""
        return {'status': 'success', 'message': 'Pinecone handles optimization automatically'}

    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone statistics"""
        stats = self.index.describe_index_stats()
        return {
            'backend': 'Pinecone',
            'index_name': self.index_name,
            'total_vectors': stats.get('total_vector_count', 0),
            'dimension': self.config.dimension,
            'metric': self.config.metric.value
        }


class WeaviateBackend(VectorDatabaseBackend):
    """Weaviate vector database backend"""

    def __init__(self, config: IndexConfig, url: str = "http://localhost:8080", class_name: str = "Document"):
        super().__init__(config)
        self.url = url
        self.class_name = class_name
        self.client = None
        self.create_index()

    def create_index(self) -> bool:
        """Create Weaviate class/schema"""
        try:
            import weaviate

            self.client = weaviate.Client(self.url)

            # Check if class exists
            if not self.client.schema.exists(self.class_name):
                class_obj = {
                    "class": self.class_name,
                    "vectorizer": "none",  # We provide vectors
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"]
                        },
                        {
                            "name": "metadata",
                            "dataType": ["text"]  # Store as JSON string
                        }
                    ]
                }
                self.client.schema.create_class(class_obj)

            logger.info(f"Weaviate class created/connected: {self.class_name}")
            return True

        except ImportError:
            logger.error("Weaviate not installed. Install with: pip install weaviate-client")
            return False
        except Exception as e:
            logger.error(f"Error creating Weaviate class: {str(e)}")
            return False

    def insert(self, document: VectorDocument) -> bool:
        """Insert document into Weaviate"""
        try:
            data_object = {
                "text": document.text or "",
                "metadata": json.dumps(document.metadata)
            }

            self.client.data_object.create(
                data_object=data_object,
                class_name=self.class_name,
                uuid=document.id,
                vector=document.vector.tolist()
            )

            self.documents[document.id] = document
            return True

        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return False

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert into Weaviate"""
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                for doc in documents:
                    data_object = {
                        "text": doc.text or "",
                        "metadata": json.dumps(doc.metadata)
                    }
                    batch.add_data_object(
                        data_object=data_object,
                        class_name=self.class_name,
                        uuid=doc.id,
                        vector=doc.vector.tolist()
                    )
                    self.documents[doc.id] = doc

            return {
                'status': 'success',
                'inserted': len(documents),
                'total': len(self.documents)
            }

        except Exception as e:
            logger.error(f"Error in batch insert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in Weaviate"""
        try:
            query = self.client.query.get(self.class_name, ["text", "metadata"]) \
                .with_near_vector({"vector": query_vector.tolist()}) \
                .with_limit(top_k) \
                .with_additional(["id", "certainty"])

            # Apply filters if provided
            if filters:
                # Build where filter
                where_filter = self._build_where_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)

            results = query.do()

            search_results = []
            if 'data' in results and 'Get' in results['data']:
                objects = results['data']['Get'].get(self.class_name, [])
                for rank, obj in enumerate(objects):
                    doc_id = obj['_additional']['id']
                    certainty = obj['_additional']['certainty']

                    document = self.documents.get(doc_id)
                    if document:
                        search_results.append(SearchResult(
                            id=doc_id,
                            score=float(certainty),
                            document=document,
                            rank=rank
                        ))

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def _build_where_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter from filters dict"""
        # Simplified filter building - can be extended
        return None

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document in Weaviate"""
        try:
            data_object = {
                "text": document.text or "",
                "metadata": json.dumps(document.metadata)
            }

            self.client.data_object.update(
                data_object=data_object,
                class_name=self.class_name,
                uuid=doc_id,
                vector=document.vector.tolist()
            )

            self.documents[doc_id] = document
            return True

        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def delete(self, doc_id: str) -> bool:
        """Delete document from Weaviate"""
        try:
            self.client.data_object.delete(uuid=doc_id, class_name=self.class_name)
            if doc_id in self.documents:
                del self.documents[doc_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete from Weaviate"""
        try:
            deleted = 0
            with self.client.batch as batch:
                for doc_id in doc_ids:
                    batch.delete_objects(
                        class_name=self.class_name,
                        where={"path": ["id"], "operator": "Equal", "valueString": doc_id}
                    )
                    if doc_id in self.documents:
                        del self.documents[doc_id]
                        deleted += 1

            return {'status': 'success', 'deleted': deleted}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize Weaviate index"""
        return {'status': 'success', 'message': 'Weaviate handles optimization automatically'}

    def get_stats(self) -> Dict[str, Any]:
        """Get Weaviate statistics"""
        return {
            'backend': 'Weaviate',
            'class_name': self.class_name,
            'total_vectors': len(self.documents),
            'dimension': self.config.dimension,
            'metric': self.config.metric.value
        }


class QdrantBackend(VectorDatabaseBackend):
    """Qdrant vector database backend"""

    def __init__(self, config: IndexConfig, url: str = "http://localhost:6333", collection_name: str = "default"):
        super().__init__(config)
        self.url = url
        self.collection_name = collection_name
        self.client = None
        self.create_index()

    def create_index(self) -> bool:
        """Create Qdrant collection"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(url=self.url)

            # Map distance metric
            distance_map = {
                DistanceMetric.COSINE: Distance.COSINE,
                DistanceMetric.EUCLIDEAN: Distance.EUCLID,
                DistanceMetric.DOT_PRODUCT: Distance.DOT
            }

            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.dimension,
                        distance=distance_map.get(self.config.metric, Distance.COSINE)
                    )
                )

            logger.info(f"Qdrant collection created/connected: {self.collection_name}")
            return True

        except ImportError:
            logger.error("Qdrant not installed. Install with: pip install qdrant-client")
            return False
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            return False

    def insert(self, document: VectorDocument) -> bool:
        """Insert document into Qdrant"""
        try:
            from qdrant_client.models import PointStruct

            point = PointStruct(
                id=document.id,
                vector=document.vector.tolist(),
                payload={
                    'text': document.text,
                    'metadata': document.metadata,
                    'timestamp': document.timestamp
                }
            )

            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            self.documents[document.id] = document
            return True

        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return False

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert into Qdrant"""
        try:
            from qdrant_client.models import PointStruct

            points = [
                PointStruct(
                    id=doc.id,
                    vector=doc.vector.tolist(),
                    payload={
                        'text': doc.text,
                        'metadata': doc.metadata,
                        'timestamp': doc.timestamp
                    }
                )
                for doc in documents
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            for doc in documents:
                self.documents[doc.id] = doc

            collection_info = self.client.get_collection(self.collection_name)
            return {
                'status': 'success',
                'inserted': len(documents),
                'total': collection_info.points_count
            }

        except Exception as e:
            logger.error(f"Error in batch insert: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search in Qdrant"""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Build filter
            query_filter = None
            if filters:
                conditions = [
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
                query_filter = Filter(must=conditions)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=query_filter
            )

            search_results = []
            for rank, hit in enumerate(results):
                doc_id = hit.id
                score = hit.score

                document = self.documents.get(doc_id)
                if document:
                    search_results.append(SearchResult(
                        id=doc_id,
                        score=float(score),
                        document=document,
                        rank=rank
                    ))

            return search_results

        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return []

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document in Qdrant"""
        return self.insert(document)  # Upsert handles updates

    def delete(self, doc_id: str) -> bool:
        """Delete document from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            if doc_id in self.documents:
                del self.documents[doc_id]
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete from Qdrant"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=doc_ids
            )

            deleted = 0
            for doc_id in doc_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
                    deleted += 1

            return {'status': 'success', 'deleted': deleted}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize Qdrant index"""
        try:
            # Qdrant optimization can involve creating payload indexes
            return {'status': 'success', 'message': 'Qdrant optimization available through payload indexes'}

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get Qdrant statistics"""
        collection_info = self.client.get_collection(self.collection_name)
        return {
            'backend': 'Qdrant',
            'collection_name': self.collection_name,
            'total_vectors': collection_info.points_count,
            'dimension': self.config.dimension,
            'metric': self.config.metric.value,
            'indexed_vectors': collection_info.indexed_vectors_count
        }


class HybridSearchEngine:
    """Hybrid search combining vector similarity and keyword matching"""

    def __init__(self, vector_backend: VectorDatabaseBackend, alpha: float = 0.5):
        """
        Initialize hybrid search

        Args:
            vector_backend: Vector database backend
            alpha: Weight for vector search (1-alpha for keyword search)
        """
        self.vector_backend = vector_backend
        self.alpha = alpha
        self.keyword_index = defaultdict(set)  # Inverted index for keywords
        self._build_keyword_index()

    def _build_keyword_index(self):
        """Build keyword inverted index"""
        for doc_id, document in self.vector_backend.documents.items():
            if document.text:
                words = document.text.lower().split()
                for word in words:
                    self.keyword_index[word].add(doc_id)

    def keyword_search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """Perform keyword search using TF-IDF-like scoring"""
        query_words = query.lower().split()
        doc_scores = defaultdict(float)

        for word in query_words:
            if word in self.keyword_index:
                # Simple scoring: inverse document frequency
                idf = np.log(len(self.vector_backend.documents) / len(self.keyword_index[word]))
                for doc_id in self.keyword_index[word]:
                    doc_scores[doc_id] += idf

        # Normalize scores
        max_score = max(doc_scores.values()) if doc_scores else 1.0
        return {doc_id: score / max_score for doc_id, score in doc_scores.items()}

    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search

        Args:
            query_vector: Query embedding vector
            query_text: Query text for keyword search
            top_k: Number of results to return
            filters: Metadata filters

        Returns:
            List of search results sorted by combined score
        """
        # Get vector search results
        vector_results = self.vector_backend.search(
            query_vector=query_vector,
            top_k=top_k * 2,  # Get more results for reranking
            filters=filters
        )

        # Get keyword search scores
        keyword_scores = self.keyword_search(query_text, top_k * 2)

        # Combine scores
        combined_results = {}

        # Add vector search results
        for result in vector_results:
            vector_score = result.score
            keyword_score = keyword_scores.get(result.id, 0.0)
            combined_score = self.alpha * vector_score + (1 - self.alpha) * keyword_score

            combined_results[result.id] = SearchResult(
                id=result.id,
                score=combined_score,
                document=result.document,
                rank=0  # Will be updated after sorting
            )

        # Add any keyword-only results
        for doc_id, keyword_score in keyword_scores.items():
            if doc_id not in combined_results:
                document = self.vector_backend.documents.get(doc_id)
                if document:
                    combined_score = (1 - self.alpha) * keyword_score
                    combined_results[doc_id] = SearchResult(
                        id=doc_id,
                        score=combined_score,
                        document=document,
                        rank=0
                    )

        # Sort by combined score
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.score,
            reverse=True
        )

        # Update ranks
        for rank, result in enumerate(sorted_results[:top_k]):
            result.rank = rank

        return sorted_results[:top_k]


class VectorDatabaseManager:
    """
    Main manager class for vector database operations
    Provides unified interface across multiple backends
    """

    def __init__(
        self,
        backend_type: str = "faiss",
        config: Optional[IndexConfig] = None,
        **backend_kwargs
    ):
        """
        Initialize vector database manager

        Args:
            backend_type: Type of backend ("faiss", "chroma", "pinecone", "weaviate", "qdrant")
            config: Index configuration
            **backend_kwargs: Backend-specific arguments
        """
        self.config = config or IndexConfig()
        self.backend_type = backend_type.lower()
        self.backend = self._create_backend(**backend_kwargs)
        self.hybrid_search_engine = None

    def _create_backend(self, **kwargs) -> VectorDatabaseBackend:
        """Create appropriate backend instance"""
        backends = {
            'faiss': FAISSBackend,
            'chroma': ChromaBackend,
            'pinecone': PineconeBackend,
            'weaviate': WeaviateBackend,
            'qdrant': QdrantBackend
        }

        backend_class = backends.get(self.backend_type)
        if not backend_class:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        return backend_class(self.config, **kwargs)

    def enable_hybrid_search(self, alpha: float = 0.5):
        """Enable hybrid search functionality"""
        self.hybrid_search_engine = HybridSearchEngine(self.backend, alpha=alpha)
        logger.info("Hybrid search enabled")

    def insert(self, document: VectorDocument) -> bool:
        """Insert single document"""
        result = self.backend.insert(document)
        if result and self.hybrid_search_engine:
            self.hybrid_search_engine._build_keyword_index()
        return result

    def batch_insert(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Batch insert documents"""
        result = self.backend.batch_insert(documents)
        if result.get('status') == 'success' and self.hybrid_search_engine:
            self.hybrid_search_engine._build_keyword_index()
        return result

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors

        Args:
            query_vector: Query embedding
            top_k: Number of results
            filters: Metadata filters
            query_text: Text query for hybrid search

        Returns:
            List of search results
        """
        if query_text and self.hybrid_search_engine:
            return self.hybrid_search_engine.hybrid_search(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filters=filters
            )
        return self.backend.search(query_vector, top_k, filters)

    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.backend.get(doc_id)

    def update(self, doc_id: str, document: VectorDocument) -> bool:
        """Update document"""
        result = self.backend.update(doc_id, document)
        if result and self.hybrid_search_engine:
            self.hybrid_search_engine._build_keyword_index()
        return result

    def delete(self, doc_id: str) -> bool:
        """Delete document"""
        result = self.backend.delete(doc_id)
        if result and self.hybrid_search_engine:
            self.hybrid_search_engine._build_keyword_index()
        return result

    def batch_delete(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Batch delete documents"""
        result = self.backend.batch_delete(doc_ids)
        if result.get('status') == 'success' and self.hybrid_search_engine:
            self.hybrid_search_engine._build_keyword_index()
        return result

    def optimize_index(self) -> Dict[str, Any]:
        """Optimize index"""
        return self.backend.optimize_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return self.backend.get_stats()

    def export_index(self, filepath: str) -> bool:
        """Export index to file"""
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'config': self.config,
                    'backend_type': self.backend_type,
                    'documents': self.backend.documents
                }, f)
            logger.info(f"Index exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting index: {str(e)}")
            return False

    def import_index(self, filepath: str) -> bool:
        """Import index from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.config = data['config']
            self.backend_type = data['backend_type']

            # Recreate backend and reinsert documents
            documents = list(data['documents'].values())
            if documents:
                self.batch_insert(documents)

            logger.info(f"Index imported from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error importing index: {str(e)}")
            return False


# Example usage and utilities
def create_sample_documents(num_docs: int = 100, dimension: int = 768) -> List[VectorDocument]:
    """Create sample documents for testing"""
    documents = []
    for i in range(num_docs):
        vector = np.random.randn(dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize

        doc = VectorDocument(
            id=f"doc_{i}",
            vector=vector,
            text=f"This is sample document {i} with some text content.",
            metadata={
                'category': f"category_{i % 5}",
                'priority': i % 3,
                'source': 'sample_generator'
            }
        )
        documents.append(doc)

    return documents


def demo_faiss():
    """Demo FAISS backend"""
    print("\n=== FAISS Backend Demo ===")

    config = IndexConfig(
        index_type=IndexType.FLAT,
        dimension=768,
        metric=DistanceMetric.COSINE
    )

    manager = VectorDatabaseManager(backend_type="faiss", config=config)

    # Create sample documents
    documents = create_sample_documents(num_docs=50, dimension=768)

    # Batch insert
    result = manager.batch_insert(documents)
    print(f"Batch insert: {result}")

    # Search
    query_vector = np.random.randn(768).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = manager.search(query_vector, top_k=5)
    print(f"\nTop 5 search results:")
    for result in results:
        print(f"  - {result.id}: score={result.score:.4f}")

    # Filter search
    filtered_results = manager.search(
        query_vector,
        top_k=5,
        filters={'category': 'category_1'}
    )
    print(f"\nFiltered search (category_1): {len(filtered_results)} results")

    # Stats
    stats = manager.get_stats()
    print(f"\nIndex stats: {stats}")


def demo_hybrid_search():
    """Demo hybrid search"""
    print("\n=== Hybrid Search Demo ===")

    config = IndexConfig(dimension=768)
    manager = VectorDatabaseManager(backend_type="faiss", config=config)
    manager.enable_hybrid_search(alpha=0.7)  # 70% vector, 30% keyword

    # Create documents with meaningful text
    documents = [
        VectorDocument(
            id="doc_1",
            vector=np.random.randn(768).astype(np.float32),
            text="Machine learning is a subset of artificial intelligence",
            metadata={'topic': 'AI'}
        ),
        VectorDocument(
            id="doc_2",
            vector=np.random.randn(768).astype(np.float32),
            text="Deep learning uses neural networks with many layers",
            metadata={'topic': 'AI'}
        ),
        VectorDocument(
            id="doc_3",
            vector=np.random.randn(768).astype(np.float32),
            text="Python is a popular programming language for data science",
            metadata={'topic': 'Programming'}
        )
    ]

    # Normalize vectors
    for doc in documents:
        doc.vector = doc.vector / np.linalg.norm(doc.vector)

    manager.batch_insert(documents)

    # Hybrid search
    query_vector = np.random.randn(768).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = manager.search(
        query_vector=query_vector,
        query_text="machine learning neural networks",
        top_k=3
    )

    print(f"Hybrid search results:")
    for result in results:
        print(f"  - {result.id}: score={result.score:.4f}, text={result.document.text[:50]}...")


if __name__ == "__main__":
    print("VectorDatabases - Production-Ready Vector Database System")
    print("=" * 60)

    # Run demos
    try:
        demo_faiss()
        demo_hybrid_search()

        print("\n" + "=" * 60)
        print("Demos completed successfully!")
        print("\nSupported backends:")
        print("  - FAISS (Facebook AI Similarity Search)")
        print("  - ChromaDB")
        print("  - Pinecone")
        print("  - Weaviate")
        print("  - Qdrant")

    except Exception as e:
        logger.error(f"Demo error: {str(e)}")
        print("\nNote: Some backends require additional installation.")
        print("Install dependencies: pip install -r requirements.txt")
