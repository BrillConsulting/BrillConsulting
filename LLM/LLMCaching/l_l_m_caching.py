"""
LLMCaching
Author: BrillConsulting
Description: Production-ready response caching system for LLMs with semantic similarity,
            distributed caching, Redis integration, and comprehensive analytics.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
import logging
from collections import defaultdict
import threading
from abc import ABC, abstractmethod

try:
    import redis
    from redis.cluster import RedisCluster
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
    NDArray = np.ndarray
except ImportError:
    SEMANTIC_AVAILABLE = False
    np = None
    NDArray = Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class InvalidationStrategy(Enum):
    """Cache invalidation strategies"""
    MANUAL = "manual"  # Explicit invalidation
    TTL_BASED = "ttl_based"  # Time-based expiration
    PATTERN_BASED = "pattern_based"  # Pattern matching
    VERSIONED = "versioned"  # Version-based invalidation


@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    key: str
    value: Any
    timestamp: float
    ttl: Optional[int] = None  # seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    embedding: Optional[NDArray] = None
    tags: List[str] = field(default_factory=list)
    version: str = "v1"

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def access(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStatistics:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_requests: int = 0
    average_latency_ms: float = 0.0
    hit_rate: float = 0.0
    semantic_hit_rate: float = 0.0
    cache_size: int = 0
    memory_usage_bytes: int = 0

    def update_hit_rate(self):
        """Calculate hit rates"""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100
            self.semantic_hit_rate = (self.semantic_hits / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'semantic_hits': self.semantic_hits,
            'evictions': self.evictions,
            'invalidations': self.invalidations,
            'total_requests': self.total_requests,
            'hit_rate': round(self.hit_rate, 2),
            'semantic_hit_rate': round(self.semantic_hit_rate, 2),
            'average_latency_ms': round(self.average_latency_ms, 2),
            'cache_size': self.cache_size,
            'memory_usage_mb': round(self.memory_usage_bytes / 1024 / 1024, 2)
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry from cache"""
        pass

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> bool:
        """Store entry in cache"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all entries"""
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Get all cache keys"""
        pass


class InMemoryCacheBackend(CacheBackend):
    """In-memory cache backend with thread safety"""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[CacheEntry]:
        with self.lock:
            entry = self.cache.get(key)
            if entry and not entry.is_expired():
                entry.access()
                return entry
            elif entry:
                del self.cache[key]
            return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        with self.lock:
            if len(self.cache) >= self.max_size:
                return False
            self.cache[key] = entry
            return True

    def delete(self, key: str) -> bool:
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> bool:
        with self.lock:
            self.cache.clear()
            return True

    def keys(self) -> List[str]:
        with self.lock:
            return list(self.cache.keys())

    def size(self) -> int:
        with self.lock:
            return len(self.cache)


class RedisCacheBackend(CacheBackend):
    """Redis-based distributed cache backend"""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        cluster_mode: bool = False,
        cluster_nodes: Optional[List[Dict[str, Any]]] = None,
        prefix: str = 'llm_cache:'
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available. Install with: pip install redis")

        self.prefix = prefix

        if cluster_mode and cluster_nodes:
            self.client = RedisCluster(
                startup_nodes=cluster_nodes,
                decode_responses=False,
                password=password
            )
        else:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        data = {
            'key': entry.key,
            'value': entry.value,
            'timestamp': entry.timestamp,
            'ttl': entry.ttl,
            'access_count': entry.access_count,
            'last_accessed': entry.last_accessed,
            'tags': entry.tags,
            'version': entry.version
        }
        if entry.embedding is not None:
            data['embedding'] = entry.embedding.tolist()
        return json.dumps(data).encode('utf-8')

    def _deserialize_entry(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry"""
        obj = json.loads(data.decode('utf-8'))
        embedding = None
        if 'embedding' in obj and obj['embedding']:
            embedding = np.array(obj['embedding'])
        return CacheEntry(
            key=obj['key'],
            value=obj['value'],
            timestamp=obj['timestamp'],
            ttl=obj.get('ttl'),
            access_count=obj.get('access_count', 0),
            last_accessed=obj.get('last_accessed', time.time()),
            embedding=embedding,
            tags=obj.get('tags', []),
            version=obj.get('version', 'v1')
        )

    def get(self, key: str) -> Optional[CacheEntry]:
        full_key = f"{self.prefix}{key}"
        data = self.client.get(full_key)
        if data:
            entry = self._deserialize_entry(data)
            if not entry.is_expired():
                entry.access()
                self.set(key, entry)  # Update access metadata
                return entry
            else:
                self.delete(key)
        return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        full_key = f"{self.prefix}{key}"
        data = self._serialize_entry(entry)
        if entry.ttl:
            self.client.setex(full_key, entry.ttl, data)
        else:
            self.client.set(full_key, data)
        return True

    def delete(self, key: str) -> bool:
        full_key = f"{self.prefix}{key}"
        return bool(self.client.delete(full_key))

    def clear(self) -> bool:
        pattern = f"{self.prefix}*"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
        return True

    def keys(self) -> List[str]:
        pattern = f"{self.prefix}*"
        keys = self.client.keys(pattern)
        prefix_len = len(self.prefix)
        return [k.decode('utf-8')[prefix_len:] for k in keys]


class SemanticCacheEngine:
    """Semantic similarity-based cache matching"""

    def __init__(self, similarity_threshold: float = 0.85):
        if not SEMANTIC_AVAILABLE:
            raise ImportError("Semantic caching requires numpy and scikit-learn")

        self.similarity_threshold = similarity_threshold
        self.embeddings_cache: Dict[str, Tuple[CacheEntry, NDArray]] = {}
        self.lock = threading.RLock()

    def add_embedding(self, key: str, entry: CacheEntry, embedding: NDArray):
        """Add embedding for semantic matching"""
        with self.lock:
            entry.embedding = embedding
            self.embeddings_cache[key] = (entry, embedding)

    def find_similar(self, query_embedding: NDArray) -> Optional[Tuple[str, CacheEntry, float]]:
        """Find semantically similar cached entry"""
        with self.lock:
            if not self.embeddings_cache:
                return None

            best_match = None
            best_similarity = 0.0

            for key, (entry, embedding) in self.embeddings_cache.items():
                if entry.is_expired():
                    continue

                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0][0]

                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = (key, entry, similarity)

            return best_match

    def remove_embedding(self, key: str):
        """Remove embedding from cache"""
        with self.lock:
            self.embeddings_cache.pop(key, None)

    def clear(self):
        """Clear all embeddings"""
        with self.lock:
            self.embeddings_cache.clear()


class LLMCachingManager:
    """
    Production-ready LLM caching system with advanced features.

    Features:
    - Semantic caching with cosine similarity
    - Redis integration for distributed caching
    - TTL management with automatic expiration
    - Multiple cache invalidation strategies
    - Hit rate analytics and monitoring
    - Thread-safe operations
    - Configurable eviction policies
    """

    def __init__(
        self,
        backend: Optional[CacheBackend] = None,
        enable_semantic: bool = False,
        semantic_threshold: float = 0.85,
        default_ttl: Optional[int] = 3600,
        max_cache_size: int = 10000,
        eviction_strategy: CacheStrategy = CacheStrategy.LRU,
        invalidation_strategy: InvalidationStrategy = InvalidationStrategy.TTL_BASED
    ):
        """
        Initialize LLM Caching Manager.

        Args:
            backend: Cache backend (InMemory or Redis)
            enable_semantic: Enable semantic similarity matching
            semantic_threshold: Minimum similarity score for semantic matches
            default_ttl: Default time-to-live in seconds
            max_cache_size: Maximum number of cache entries
            eviction_strategy: Strategy for cache eviction
            invalidation_strategy: Strategy for cache invalidation
        """
        self.backend = backend or InMemoryCacheBackend(max_size=max_cache_size)
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.eviction_strategy = eviction_strategy
        self.invalidation_strategy = invalidation_strategy
        self.statistics = CacheStatistics()
        self.latency_samples: List[float] = []

        # Semantic caching
        self.enable_semantic = enable_semantic and SEMANTIC_AVAILABLE
        self.semantic_engine = None
        if self.enable_semantic:
            self.semantic_engine = SemanticCacheEngine(similarity_threshold=semantic_threshold)

        # Version tracking for invalidation
        self.version_registry: Dict[str, str] = {}
        self.lock = threading.RLock()

        logger.info(f"LLM Caching Manager initialized with {type(self.backend).__name__}")

    def _generate_key(self, prompt: str, model: str = "default", **kwargs) -> str:
        """Generate cache key from prompt and parameters"""
        key_data = {
            'prompt': prompt,
            'model': model,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _evict_if_needed(self):
        """Evict entries based on eviction strategy"""
        keys = self.backend.keys()
        if len(keys) < self.max_cache_size:
            return

        entries_to_evict = []

        for key in keys:
            entry = self.backend.get(key)
            if entry:
                entries_to_evict.append((key, entry))

        if not entries_to_evict:
            return

        # Sort based on eviction strategy
        if self.eviction_strategy == CacheStrategy.LRU:
            entries_to_evict.sort(key=lambda x: x[1].last_accessed)
        elif self.eviction_strategy == CacheStrategy.LFU:
            entries_to_evict.sort(key=lambda x: x[1].access_count)
        elif self.eviction_strategy == CacheStrategy.FIFO:
            entries_to_evict.sort(key=lambda x: x[1].timestamp)
        elif self.eviction_strategy == CacheStrategy.TTL:
            entries_to_evict.sort(key=lambda x: x[1].timestamp)

        # Evict oldest/least used entries
        num_to_evict = len(entries_to_evict) - self.max_cache_size + 1
        for i in range(num_to_evict):
            key = entries_to_evict[i][0]
            self.backend.delete(key)
            if self.semantic_engine:
                self.semantic_engine.remove_embedding(key)
            self.statistics.evictions += 1

    def get(
        self,
        prompt: str,
        model: str = "default",
        embedding: Optional[NDArray] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Retrieve cached response.

        Args:
            prompt: Input prompt
            model: Model identifier
            embedding: Optional embedding for semantic matching
            **kwargs: Additional parameters for cache key generation

        Returns:
            Cached response or None if not found
        """
        start_time = time.time()

        with self.lock:
            self.statistics.total_requests += 1

            # Try exact match
            key = self._generate_key(prompt, model, **kwargs)
            entry = self.backend.get(key)

            if entry:
                self.statistics.hits += 1
                latency = (time.time() - start_time) * 1000
                self._update_latency(latency)
                self.statistics.update_hit_rate()
                logger.debug(f"Cache hit for key: {key[:16]}...")
                return entry.value

            # Try semantic match if enabled
            if self.enable_semantic and embedding is not None and self.semantic_engine:
                match = self.semantic_engine.find_similar(embedding)
                if match:
                    _, entry, similarity = match
                    self.statistics.hits += 1
                    self.statistics.semantic_hits += 1
                    latency = (time.time() - start_time) * 1000
                    self._update_latency(latency)
                    self.statistics.update_hit_rate()
                    logger.debug(f"Semantic cache hit with similarity: {similarity:.3f}")
                    return entry.value

            # Cache miss
            self.statistics.misses += 1
            latency = (time.time() - start_time) * 1000
            self._update_latency(latency)
            self.statistics.update_hit_rate()
            logger.debug(f"Cache miss for prompt: {prompt[:50]}...")
            return None

    def set(
        self,
        prompt: str,
        response: Any,
        model: str = "default",
        ttl: Optional[int] = None,
        embedding: Optional[NDArray] = None,
        tags: Optional[List[str]] = None,
        version: str = "v1",
        **kwargs
    ) -> bool:
        """
        Store response in cache.

        Args:
            prompt: Input prompt
            response: Response to cache
            model: Model identifier
            ttl: Time-to-live in seconds
            embedding: Optional embedding for semantic matching
            tags: Optional tags for organization
            version: Version identifier
            **kwargs: Additional parameters for cache key generation

        Returns:
            True if cached successfully
        """
        with self.lock:
            key = self._generate_key(prompt, model, **kwargs)

            # Evict if needed
            self._evict_if_needed()

            entry = CacheEntry(
                key=key,
                value=response,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                tags=tags or [],
                version=version
            )

            success = self.backend.set(key, entry)

            if success and self.enable_semantic and embedding is not None and self.semantic_engine:
                self.semantic_engine.add_embedding(key, entry, embedding)

            self._update_cache_stats()
            logger.debug(f"Cached response for key: {key[:16]}...")
            return success

    def invalidate(
        self,
        pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
        version: Optional[str] = None,
        keys: Optional[List[str]] = None
    ) -> int:
        """
        Invalidate cache entries based on various criteria.

        Args:
            pattern: Key pattern to match (supports wildcards)
            tags: List of tags to match
            version: Version to invalidate
            keys: Specific keys to invalidate

        Returns:
            Number of invalidated entries
        """
        with self.lock:
            invalidated = 0

            if keys:
                for key in keys:
                    if self.backend.delete(key):
                        invalidated += 1
                        if self.semantic_engine:
                            self.semantic_engine.remove_embedding(key)
            else:
                all_keys = self.backend.keys()
                for key in all_keys:
                    should_invalidate = False
                    entry = self.backend.get(key)

                    if not entry:
                        continue

                    if pattern and self._match_pattern(key, pattern):
                        should_invalidate = True

                    if tags and any(tag in entry.tags for tag in tags):
                        should_invalidate = True

                    if version and entry.version == version:
                        should_invalidate = True

                    if should_invalidate:
                        self.backend.delete(key)
                        if self.semantic_engine:
                            self.semantic_engine.remove_embedding(key)
                        invalidated += 1

            self.statistics.invalidations += invalidated
            self._update_cache_stats()
            logger.info(f"Invalidated {invalidated} cache entries")
            return invalidated

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """Simple pattern matching with wildcards"""
        if '*' not in pattern:
            return key == pattern

        parts = pattern.split('*')
        if not key.startswith(parts[0]):
            return False

        pos = len(parts[0])
        for part in parts[1:-1]:
            idx = key.find(part, pos)
            if idx == -1:
                return False
            pos = idx + len(part)

        if parts[-1] and not key.endswith(parts[-1]):
            return False

        return True

    def clear(self) -> bool:
        """Clear all cache entries"""
        with self.lock:
            success = self.backend.clear()
            if success and self.semantic_engine:
                self.semantic_engine.clear()
            self.statistics = CacheStatistics()
            self.latency_samples.clear()
            logger.info("Cache cleared")
            return success

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self.lock:
            self._update_cache_stats()
            return self.statistics.to_dict()

    def _update_latency(self, latency_ms: float):
        """Update average latency"""
        self.latency_samples.append(latency_ms)
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]
        self.statistics.average_latency_ms = sum(self.latency_samples) / len(self.latency_samples)

    def _update_cache_stats(self):
        """Update cache size statistics"""
        keys = self.backend.keys()
        self.statistics.cache_size = len(keys)

        # Estimate memory usage (rough approximation)
        total_size = 0
        for key in keys[:100]:  # Sample first 100 for performance
            entry = self.backend.get(key)
            if entry:
                total_size += len(str(entry.value).encode('utf-8'))

        if len(keys) > 0:
            avg_size = total_size / min(len(keys), 100)
            self.statistics.memory_usage_bytes = int(avg_size * len(keys))

    def get_top_entries(self, limit: int = 10, sort_by: str = "access_count") -> List[Dict[str, Any]]:
        """
        Get top cache entries by various metrics.

        Args:
            limit: Maximum number of entries to return
            sort_by: Metric to sort by (access_count, last_accessed, timestamp)

        Returns:
            List of top entries with metadata
        """
        with self.lock:
            entries = []
            for key in self.backend.keys():
                entry = self.backend.get(key)
                if entry:
                    entries.append({
                        'key': key[:16] + '...',
                        'access_count': entry.access_count,
                        'last_accessed': datetime.fromtimestamp(entry.last_accessed).isoformat(),
                        'created': datetime.fromtimestamp(entry.timestamp).isoformat(),
                        'ttl': entry.ttl,
                        'tags': entry.tags,
                        'version': entry.version
                    })

            entries.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            return entries[:limit]

    def export_stats(self, filepath: str):
        """Export statistics to JSON file"""
        stats = self.get_statistics()
        stats['timestamp'] = datetime.now().isoformat()
        stats['top_entries'] = self.get_top_entries(limit=20)

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics exported to {filepath}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache system"""
        try:
            # Test basic operations
            test_key = "_health_check_"
            test_value = {"status": "ok", "timestamp": time.time()}

            self.backend.set(test_key, CacheEntry(
                key=test_key,
                value=test_value,
                timestamp=time.time(),
                ttl=60
            ))

            retrieved = self.backend.get(test_key)
            self.backend.delete(test_key)

            return {
                'status': 'healthy' if retrieved else 'degraded',
                'backend': type(self.backend).__name__,
                'semantic_enabled': self.enable_semantic,
                'cache_size': len(self.backend.keys()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


def create_redis_cache_manager(
    host: str = 'localhost',
    port: int = 6379,
    password: Optional[str] = None,
    enable_semantic: bool = True,
    **kwargs
) -> LLMCachingManager:
    """
    Factory function to create Redis-backed cache manager.

    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        enable_semantic: Enable semantic caching
        **kwargs: Additional arguments for LLMCachingManager

    Returns:
        Configured LLMCachingManager instance
    """
    backend = RedisCacheBackend(host=host, port=port, password=password)
    return LLMCachingManager(backend=backend, enable_semantic=enable_semantic, **kwargs)


def create_in_memory_cache_manager(
    max_size: int = 10000,
    enable_semantic: bool = True,
    **kwargs
) -> LLMCachingManager:
    """
    Factory function to create in-memory cache manager.

    Args:
        max_size: Maximum cache size
        enable_semantic: Enable semantic caching
        **kwargs: Additional arguments for LLMCachingManager

    Returns:
        Configured LLMCachingManager instance
    """
    backend = InMemoryCacheBackend(max_size=max_size)
    return LLMCachingManager(backend=backend, enable_semantic=enable_semantic, **kwargs)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("LLM Caching System - Production Ready")
    print("=" * 80)

    # Create in-memory cache manager
    manager = create_in_memory_cache_manager(
        max_size=1000,
        enable_semantic=False,
        default_ttl=3600
    )

    # Example: Cache and retrieve responses
    print("\n1. Caching LLM responses...")
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate fibonacci numbers"
    ]

    for i, prompt in enumerate(prompts):
        response = f"Sample response for: {prompt}"
        manager.set(
            prompt=prompt,
            response=response,
            model="gpt-4",
            tags=["example", f"batch_{i}"]
        )
        print(f"  Cached: {prompt[:50]}...")

    # Retrieve cached responses
    print("\n2. Retrieving cached responses...")
    for prompt in prompts:
        cached = manager.get(prompt, model="gpt-4")
        if cached:
            print(f"  Hit: {prompt[:50]}...")
        else:
            print(f"  Miss: {prompt[:50]}...")

    # Display statistics
    print("\n3. Cache Statistics:")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Health check
    print("\n4. Health Check:")
    health = manager.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")

    # Top entries
    print("\n5. Top Cache Entries:")
    top = manager.get_top_entries(limit=5)
    for entry in top:
        print(f"  Key: {entry['key']}, Accesses: {entry['access_count']}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
