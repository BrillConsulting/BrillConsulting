"""
Cache Optimization Framework
=============================

KV cache optimization with PagedAttention and prefix caching

Author: Brill Consulting
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time


class EvictionPolicy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"


@dataclass
class CacheBlock:
    """Memory block in cache."""
    block_id: int
    size_mb: int
    last_accessed: float
    access_count: int
    data: Optional[Any] = None


class PagedAttentionEngine:
    """PagedAttention for efficient KV cache management."""

    def __init__(
        self,
        model: str,
        block_size: int = 16,
        gpu_memory_utilization: float = 0.9
    ):
        """Initialize PagedAttention engine."""
        self.model = model
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.blocks: List[CacheBlock] = []
        self.memory_saved = 0.0

        print(f"📄 PagedAttention Engine initialized")
        print(f"   Model: {model}")
        print(f"   Block size: {block_size}")
        print(f"   GPU memory target: {gpu_memory_utilization:.0%}")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100
    ) -> List[str]:
        """Generate with paged attention."""
        print(f"\n🔄 Generating with PagedAttention")
        print(f"   Batch size: {len(prompts)}")
        print(f"   Max tokens: {max_tokens}")

        # Allocate pages
        num_pages = (len(prompts) * max_tokens) // self.block_size
        print(f"   Pages allocated: {num_pages}")

        # Simulate generation
        outputs = [f"Output for: {p[:30]}..." for p in prompts]

        # Calculate memory savings
        standard_memory = len(prompts) * max_tokens * 0.1  # MB
        paged_memory = num_pages * self.block_size * 0.05  # MB
        self.memory_saved = (standard_memory - paged_memory) / standard_memory

        print(f"   ✓ Memory saved: {self.memory_saved:.1%}")

        return outputs

    @property
    def memory_savings(self) -> float:
        """Get memory savings percentage."""
        return self.memory_saved


class PrefixCache:
    """Prefix caching for reusable prompts."""

    def __init__(
        self,
        max_cache_size_gb: float = 4,
        eviction_policy: str = "lru"
    ):
        """Initialize prefix cache."""
        self.max_cache_size_gb = max_cache_size_gb
        self.eviction_policy = EvictionPolicy(eviction_policy)
        self.cache: Dict[str, Dict] = {}
        self.hit_count = 0
        self.miss_count = 0

        print(f"💾 Prefix Cache initialized")
        print(f"   Max size: {max_cache_size_gb}GB")
        print(f"   Eviction: {eviction_policy}")

    def add_prefix(self, prefix: str) -> str:
        """Add prefix to cache."""
        prefix_id = f"prefix_{len(self.cache)}"

        self.cache[prefix_id] = {
            "text": prefix,
            "computed_kv": f"<KV cache for: {prefix[:50]}>",
            "timestamp": time.time(),
            "access_count": 0
        }

        print(f"   ✓ Cached prefix: {prefix_id}")
        return prefix_id

    def generate_with_prefix(
        self,
        prefix_id: str,
        prompt: str,
        max_tokens: int = 100
    ) -> str:
        """Generate reusing cached prefix."""
        if prefix_id in self.cache:
            # Cache hit
            self.hit_count += 1
            self.cache[prefix_id]["access_count"] += 1
            self.cache[prefix_id]["timestamp"] = time.time()

            print(f"   ✅ Cache HIT for {prefix_id}")

            return f"[Using cached prefix] {prompt} -> Generated output"
        else:
            # Cache miss
            self.miss_count += 1
            print(f"   ❌ Cache MISS for {prefix_id}")
            return f"[No cache] {prompt} -> Generated output"

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def evict_oldest(self) -> None:
        """Evict based on policy."""
        if not self.cache:
            return

        if self.eviction_policy == EvictionPolicy.LRU:
            # Evict least recently used
            oldest = min(
                self.cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            del self.cache[oldest[0]]
            print(f"   🗑️  Evicted (LRU): {oldest[0]}")


class MemoryManager:
    """GPU memory pool management."""

    def __init__(
        self,
        total_gpu_memory_gb: float = 80,
        reserved_memory_gb: float = 5
    ):
        """Initialize memory manager."""
        self.total_memory = total_gpu_memory_gb
        self.reserved = reserved_memory_gb
        self.available = total_gpu_memory_gb - reserved_memory_gb
        self.pools: Dict[str, Dict] = {}

        print(f"🧠 Memory Manager initialized")
        print(f"   Total: {total_gpu_memory_gb}GB")
        print(f"   Available: {self.available}GB")

    def allocate_pool(
        self,
        name: str,
        size_gb: float,
        block_size_mb: int = 256
    ) -> Dict[str, Any]:
        """Allocate memory pool."""
        if size_gb > self.available:
            raise ValueError(f"Insufficient memory: {size_gb}GB requested")

        pool = {
            "name": name,
            "size_gb": size_gb,
            "block_size_mb": block_size_mb,
            "num_blocks": int((size_gb * 1024) / block_size_mb),
            "used_blocks": 0,
            "allocated_time": datetime.now().isoformat()
        }

        self.pools[name] = pool
        self.available -= size_gb

        print(f"   ✓ Allocated pool: {name}")
        print(f"      Size: {size_gb}GB")
        print(f"      Blocks: {pool['num_blocks']}")

        return pool

    def get_utilization(self, pool_name: str) -> float:
        """Get pool utilization."""
        pool = self.pools.get(pool_name)
        if not pool:
            return 0.0

        # Simulate utilization
        return pool["used_blocks"] / pool["num_blocks"]

    def defragment(self, pool_name: str) -> None:
        """Defragment memory pool."""
        print(f"\n🔧 Defragmenting pool: {pool_name}")
        time.sleep(0.1)  # Simulate defrag
        print(f"   ✓ Defragmentation complete")


def demo():
    """Demonstrate cache optimization."""
    print("=" * 60)
    print("Cache Optimization Framework Demo")
    print("=" * 60)

    # PagedAttention
    print(f"\n{'='*60}")
    print("PagedAttention Engine")
    print(f"{'='*60}")

    engine = PagedAttentionEngine(
        model="meta-llama/Llama-2-7b-hf",
        block_size=16,
        gpu_memory_utilization=0.9
    )

    prompts = ["Explain quantum computing"] * 10
    outputs = engine.generate(prompts, max_tokens=200)

    print(f"\n   Total memory savings: {engine.memory_savings:.1%}")

    # Prefix Caching
    print(f"\n{'='*60}")
    print("Prefix Caching")
    print(f"{'='*60}")

    cache = PrefixCache(max_cache_size_gb=4, eviction_policy="lru")

    # Add system prompt
    system_prompt = "You are a helpful AI assistant."
    prefix_id = cache.add_prefix(system_prompt)

    # Reuse prefix
    queries = [
        "What is AI?",
        "Explain machine learning",
        "Tell me about neural networks"
    ]

    for query in queries:
        output = cache.generate_with_prefix(prefix_id, query, max_tokens=100)

    print(f"\n   Cache hit rate: {cache.get_hit_rate():.1%}")

    # Memory Management
    print(f"\n{'='*60}")
    print("Memory Management")
    print(f"{'='*60}")

    manager = MemoryManager(total_gpu_memory_gb=80, reserved_memory_gb=5)

    kv_pool = manager.allocate_pool(
        name="kv_cache",
        size_gb=40,
        block_size_mb=256
    )

    # Simulate usage
    kv_pool["used_blocks"] = int(kv_pool["num_blocks"] * 0.75)

    utilization = manager.get_utilization("kv_cache")
    print(f"\n   KV cache utilization: {utilization:.1%}")

    manager.defragment("kv_cache")


if __name__ == "__main__":
    demo()
