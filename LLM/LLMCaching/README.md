# LLMCaching

Production-ready response caching system for Large Language Models (LLMs) with advanced features including semantic similarity matching, distributed caching, and comprehensive analytics.

**Author:** BrillConsulting

## Overview

LLMCaching is a sophisticated caching solution designed to optimize LLM applications by reducing redundant API calls, improving response times, and lowering costs. The system supports both in-memory and distributed Redis-backed caching with intelligent features like semantic similarity matching and configurable eviction policies.

## Features

### Core Capabilities

- **Semantic Caching**: Match similar queries using cosine similarity on embeddings
- **Distributed Caching**: Redis integration with cluster support for scalable deployments
- **TTL Management**: Automatic expiration of cached entries with configurable time-to-live
- **Cache Invalidation**: Multiple strategies including pattern-based, tag-based, and version-based invalidation
- **Hit Rate Analytics**: Comprehensive metrics and monitoring for cache performance
- **Thread Safety**: Full thread-safe operations for concurrent environments
- **Eviction Policies**: LRU, LFU, FIFO, and TTL-based eviction strategies
- **Export Capabilities**: JSON export of statistics and performance metrics

### Backend Support

- **In-Memory Backend**: Fast, thread-safe local caching with configurable size limits
- **Redis Backend**: Production-grade distributed caching with persistence
- **Redis Cluster**: Support for horizontal scaling across multiple Redis nodes

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For Redis support:
```bash
pip install redis
```

For semantic caching:
```bash
pip install scikit-learn
```

## Quick Start

### Basic Usage (In-Memory)

```python
from l_l_m_caching import create_in_memory_cache_manager

# Initialize cache manager
cache = create_in_memory_cache_manager(
    max_size=10000,
    default_ttl=3600
)

# Cache a response
cache.set(
    prompt="What is machine learning?",
    response="Machine learning is...",
    model="gpt-4"
)

# Retrieve cached response
result = cache.get(
    prompt="What is machine learning?",
    model="gpt-4"
)

print(result)  # "Machine learning is..."
```

### Redis Backend

```python
from l_l_m_caching import create_redis_cache_manager

# Initialize Redis-backed cache
cache = create_redis_cache_manager(
    host='localhost',
    port=6379,
    password='your-password',
    default_ttl=3600
)

# Use same API as in-memory cache
cache.set(prompt="...", response="...", model="gpt-4")
result = cache.get(prompt="...", model="gpt-4")
```

### Semantic Caching

```python
import numpy as np
from l_l_m_caching import create_in_memory_cache_manager

# Initialize with semantic caching enabled
cache = create_in_memory_cache_manager(
    enable_semantic=True,
    semantic_threshold=0.85
)

# Cache with embeddings
embedding1 = np.random.rand(768)  # Your embedding model output
cache.set(
    prompt="What is AI?",
    response="Artificial Intelligence is...",
    model="gpt-4",
    embedding=embedding1
)

# Query with similar embedding
embedding2 = np.random.rand(768)  # Similar question embedding
result = cache.get(
    prompt="What does AI mean?",
    model="gpt-4",
    embedding=embedding2
)
# Will return cached response if similarity >= 0.85
```

## Advanced Usage

### Custom Configuration

```python
from l_l_m_caching import (
    LLMCachingManager,
    InMemoryCacheBackend,
    CacheStrategy,
    InvalidationStrategy
)

# Create custom backend
backend = InMemoryCacheBackend(max_size=5000)

# Initialize with custom settings
cache = LLMCachingManager(
    backend=backend,
    enable_semantic=True,
    semantic_threshold=0.90,
    default_ttl=7200,
    max_cache_size=5000,
    eviction_strategy=CacheStrategy.LRU,
    invalidation_strategy=InvalidationStrategy.TTL_BASED
)
```

### Cache Invalidation

```python
# Invalidate by pattern
cache.invalidate(pattern="user_*")

# Invalidate by tags
cache.set(
    prompt="...",
    response="...",
    tags=["user:123", "category:tech"]
)
cache.invalidate(tags=["user:123"])

# Invalidate by version
cache.set(
    prompt="...",
    response="...",
    version="v1"
)
cache.invalidate(version="v1")

# Invalidate specific keys
cache.invalidate(keys=["key1", "key2"])
```

### Analytics and Monitoring

```python
# Get statistics
stats = cache.get_statistics()
print(f"Hit Rate: {stats['hit_rate']}%")
print(f"Total Requests: {stats['total_requests']}")
print(f"Cache Size: {stats['cache_size']}")
print(f"Average Latency: {stats['average_latency_ms']}ms")

# Get top entries
top = cache.get_top_entries(limit=10, sort_by="access_count")
for entry in top:
    print(f"{entry['key']}: {entry['access_count']} accesses")

# Export statistics
cache.export_stats('/path/to/stats.json')

# Health check
health = cache.health_check()
print(f"Status: {health['status']}")
```

### Redis Cluster Configuration

```python
from l_l_m_caching import LLMCachingManager, RedisCacheBackend

# Configure Redis cluster
backend = RedisCacheBackend(
    cluster_mode=True,
    cluster_nodes=[
        {"host": "node1.redis.local", "port": 6379},
        {"host": "node2.redis.local", "port": 6379},
        {"host": "node3.redis.local", "port": 6379}
    ],
    password="your-password"
)

cache = LLMCachingManager(backend=backend)
```

## API Reference

### LLMCachingManager

#### Methods

**`get(prompt, model='default', embedding=None, **kwargs)`**
- Retrieve cached response
- Returns cached value or None

**`set(prompt, response, model='default', ttl=None, embedding=None, tags=None, version='v1', **kwargs)`**
- Store response in cache
- Returns True if successful

**`invalidate(pattern=None, tags=None, version=None, keys=None)`**
- Invalidate cache entries
- Returns number of invalidated entries

**`clear()`**
- Clear all cache entries
- Returns True if successful

**`get_statistics()`**
- Get comprehensive cache statistics
- Returns dictionary with metrics

**`get_top_entries(limit=10, sort_by='access_count')`**
- Get top cache entries by metric
- Returns list of entry metadata

**`export_stats(filepath)`**
- Export statistics to JSON file

**`health_check()`**
- Perform health check
- Returns status dictionary

### Cache Strategies

**CacheStrategy Enum:**
- `LRU`: Least Recently Used
- `LFU`: Least Frequently Used
- `FIFO`: First In First Out
- `TTL`: Time To Live based

**InvalidationStrategy Enum:**
- `MANUAL`: Explicit invalidation
- `TTL_BASED`: Time-based expiration
- `PATTERN_BASED`: Pattern matching
- `VERSIONED`: Version-based invalidation

## Architecture

### Components

1. **CacheBackend**: Abstract interface for storage backends
   - `InMemoryCacheBackend`: Local thread-safe cache
   - `RedisCacheBackend`: Distributed Redis cache

2. **SemanticCacheEngine**: Handles embedding-based similarity matching

3. **CacheEntry**: Data structure for cached items with metadata

4. **CacheStatistics**: Tracks performance metrics

5. **LLMCachingManager**: Main orchestrator coordinating all components

### Data Flow

```
Request → Cache Key Generation → Exact Match Check → Semantic Match (if enabled) → Backend Query
                                         ↓ Hit                    ↓ Hit              ↓ Miss
                                    Return Value            Return Value         Return None

Store → Cache Key Generation → Eviction Check → Entry Creation → Backend Storage → Semantic Index (if enabled)
```

## Performance Considerations

### Memory Usage
- In-memory backend: ~1-5KB per cached entry (depends on response size)
- Semantic caching: Additional 3-6KB per entry for embeddings

### Latency
- In-memory cache hit: < 1ms
- Redis cache hit: 1-5ms (local) / 5-20ms (network)
- Semantic matching: 1-10ms (depends on cache size)

### Scalability
- In-memory: Single process, 10K-100K entries typical
- Redis: Multi-process/distributed, millions of entries
- Redis Cluster: Horizontal scaling across nodes

## Best Practices

1. **Choose Appropriate TTL**: Balance freshness vs hit rate
   - Short-lived data: 300-900 seconds
   - Stable data: 3600-86400 seconds

2. **Use Semantic Caching Wisely**:
   - Enable for query-heavy applications
   - Adjust threshold based on accuracy requirements (0.80-0.95)

3. **Monitor Hit Rates**:
   - Target 60%+ hit rate for cost savings
   - Export stats regularly for analysis

4. **Tag Organization**:
   - Use hierarchical tags: `user:123`, `category:tech`
   - Enable bulk invalidation by user/category

5. **Redis Configuration**:
   - Use connection pooling for high-traffic applications
   - Configure maxmemory-policy in Redis (e.g., allkeys-lru)
   - Enable persistence for production deployments

## Troubleshooting

### Low Hit Rate
- Check TTL settings (may be too short)
- Verify cache key generation (parameters should be consistent)
- Enable semantic caching for similar queries

### High Memory Usage
- Reduce max_cache_size
- Decrease TTL values
- Implement more aggressive eviction policy

### Redis Connection Issues
- Verify host/port configuration
- Check network connectivity
- Ensure Redis server is running
- Verify authentication credentials

## Examples

See the `__main__` block in `l_l_m_caching.py` for complete working examples.

## Testing

```python
# Run the example
python l_l_m_caching.py

# Expected output shows:
# - Caching operations
# - Cache hits/misses
# - Statistics
# - Health check results
# - Top entries
```

## Production Deployment

### Docker Compose Example

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  app:
    build: .
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis

volumes:
  redis_data:
```

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password
REDIS_DB=0

# Cache Configuration
CACHE_MAX_SIZE=10000
CACHE_TTL=3600
CACHE_EVICTION_STRATEGY=lru
ENABLE_SEMANTIC_CACHE=true
SEMANTIC_THRESHOLD=0.85
```

## License

Copyright BrillConsulting. All rights reserved.

## Support

For issues, questions, or contributions, please contact BrillConsulting.

## Changelog

### Version 1.0.0 (Production-Ready)
- Complete rewrite with production features
- Added semantic caching with cosine similarity
- Implemented Redis backend with cluster support
- Added comprehensive analytics and monitoring
- Thread-safe operations throughout
- Multiple eviction and invalidation strategies
- Health check and export capabilities
- Full documentation and examples
