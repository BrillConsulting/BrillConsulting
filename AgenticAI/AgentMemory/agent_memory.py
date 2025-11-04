"""
Agent Memory System
===================

Long-term and short-term memory for AI agents:
- Short-term working memory
- Long-term episodic memory
- Semantic memory (knowledge base)
- Memory consolidation
- Retrieval with relevance scoring

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import json


class MemoryItem:
    """Individual memory item."""

    def __init__(self, content: Any, memory_type: str, metadata: Optional[Dict] = None):
        self.id = f"mem_{datetime.now().timestamp()}"
        self.content = content
        self.type = memory_type
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed = self.timestamp
        self.importance = 1.0

    def access(self):
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "importance": self.importance
        }


class ShortTermMemory:
    """Working memory with limited capacity."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, content: Any, metadata: Optional[Dict] = None):
        """Add item to short-term memory."""
        item = MemoryItem(content, "short_term", metadata)
        self.memory.append(item)
        return item

    def get_recent(self, n: int = 5) -> List[MemoryItem]:
        """Get n most recent items."""
        return list(self.memory)[-n:]

    def clear(self):
        """Clear short-term memory."""
        self.memory.clear()

    def get_all(self) -> List[MemoryItem]:
        """Get all items in short-term memory."""
        return list(self.memory)


class LongTermMemory:
    """Persistent long-term memory."""

    def __init__(self):
        self.episodic_memory = []  # Events and experiences
        self.semantic_memory = {}  # Facts and knowledge

    def add_episode(self, content: Any, metadata: Optional[Dict] = None) -> MemoryItem:
        """Store episodic memory (events, experiences)."""
        item = MemoryItem(content, "episodic", metadata)
        self.episodic_memory.append(item)
        return item

    def add_knowledge(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store semantic memory (facts, knowledge)."""
        item = MemoryItem(value, "semantic", metadata)
        self.semantic_memory[key] = item
        return item

    def get_knowledge(self, key: str) -> Optional[MemoryItem]:
        """Retrieve knowledge by key."""
        item = self.semantic_memory.get(key)
        if item:
            item.access()
        return item

    def search_episodes(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """Search episodic memory (simplified text search)."""
        results = []
        query_lower = query.lower()

        for item in self.episodic_memory:
            content_str = str(item.content).lower()
            if query_lower in content_str:
                item.access()
                results.append(item)
                if len(results) >= limit:
                    break

        return results

    def get_recent_episodes(self, n: int = 10) -> List[MemoryItem]:
        """Get recent episodic memories."""
        return sorted(self.episodic_memory, key=lambda x: x.timestamp, reverse=True)[:n]

    def prune_old_memories(self, days: int = 30):
        """Remove old, infrequently accessed memories."""
        cutoff_date = datetime.now() - timedelta(days=days)

        original_count = len(self.episodic_memory)
        self.episodic_memory = [
            m for m in self.episodic_memory
            if m.last_accessed > cutoff_date or m.importance > 5.0
        ]

        pruned = original_count - len(self.episodic_memory)
        return pruned


class AgentMemory:
    """Comprehensive memory system for AI agents."""

    def __init__(self, short_term_capacity: int = 10):
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory()
        self.consolidation_threshold = 3  # Access count for consolidation

    def remember(self, content: Any, memory_type: str = "auto", metadata: Optional[Dict] = None):
        """Store memory (automatically determines type if 'auto')."""
        if memory_type == "auto":
            # Simple heuristic: important or repeated info goes to long-term
            is_important = metadata and metadata.get("important", False)
            memory_type = "long_term" if is_important else "short_term"

        if memory_type == "short_term":
            item = self.short_term.add(content, metadata)
            print(f"💭 Stored in short-term memory: {content}")
        else:
            item = self.long_term.add_episode(content, metadata)
            print(f"🧠 Stored in long-term memory: {content}")

        return item

    def store_knowledge(self, key: str, value: Any, metadata: Optional[Dict] = None):
        """Store factual knowledge."""
        item = self.long_term.add_knowledge(key, value, metadata)
        print(f"📚 Knowledge stored: {key} = {value}")
        return item

    def recall(self, query: str, memory_type: str = "all", limit: int = 5) -> List[MemoryItem]:
        """Recall memories based on query."""
        results = []

        if memory_type in ["all", "short_term"]:
            # Search short-term memory
            for item in self.short_term.get_all():
                if query.lower() in str(item.content).lower():
                    results.append(item)

        if memory_type in ["all", "long_term"]:
            # Search long-term memory
            results.extend(self.long_term.search_episodes(query, limit))

        # Sort by relevance (simplified: by access count and recency)
        results.sort(key=lambda x: (x.access_count, x.timestamp), reverse=True)

        print(f"🔍 Recalled {len(results[:limit])} memories for: {query}")
        return results[:limit]

    def consolidate(self):
        """Move frequently accessed short-term memories to long-term."""
        print("\n🔄 Consolidating memories...")

        consolidated = 0
        for item in self.short_term.get_all():
            if item.access_count >= self.consolidation_threshold:
                # Promote to long-term memory
                self.long_term.add_episode(
                    item.content,
                    {**item.metadata, "consolidated": True}
                )
                consolidated += 1

        if consolidated > 0:
            print(f"✓ Consolidated {consolidated} memories to long-term storage")
        else:
            print("✓ No memories met consolidation threshold")

        return consolidated

    def get_context(self, n: int = 5) -> List[Dict]:
        """Get recent context for agent decision making."""
        recent_short_term = self.short_term.get_recent(n)
        recent_long_term = self.long_term.get_recent_episodes(n)

        context = []

        # Add short-term memories
        for item in recent_short_term:
            context.append({
                "type": "short_term",
                "content": item.content,
                "timestamp": item.timestamp.isoformat()
            })

        # Add recent long-term memories
        for item in recent_long_term[:n]:
            context.append({
                "type": "long_term",
                "content": item.content,
                "timestamp": item.timestamp.isoformat()
            })

        return context[:n]

    def get_statistics(self) -> Dict:
        """Get memory system statistics."""
        return {
            "short_term_count": len(self.short_term.get_all()),
            "short_term_capacity": self.short_term.capacity,
            "long_term_episodes": len(self.long_term.episodic_memory),
            "semantic_knowledge": len(self.long_term.semantic_memory),
            "total_memories": len(self.short_term.get_all()) + len(self.long_term.episodic_memory)
        }

    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term.clear()
        print("✓ Short-term memory cleared")

    def prune(self, days: int = 30):
        """Prune old memories."""
        pruned = self.long_term.prune_old_memories(days)
        print(f"✓ Pruned {pruned} old memories (older than {days} days)")
        return pruned


def demo():
    """Demo agent memory system."""
    print("Agent Memory System Demo")
    print("=" * 60)

    memory = AgentMemory(short_term_capacity=5)

    # 1. Store short-term memories
    print("\n1. Short-Term Memory")
    print("-" * 60)

    memory.remember("User asked about weather", "short_term")
    memory.remember("User location: San Francisco", "short_term")
    memory.remember("Current conversation topic: AI agents", "short_term")

    # 2. Store long-term memories
    print("\n2. Long-Term Memory")
    print("-" * 60)

    memory.remember(
        "User prefers detailed technical explanations",
        "long_term",
        {"important": True}
    )
    memory.remember(
        "Previous project: Built chatbot in Q3 2024",
        "long_term",
        {"category": "project_history"}
    )

    # 3. Store knowledge
    print("\n3. Semantic Knowledge")
    print("-" * 60)

    memory.store_knowledge("user_name", "Alice")
    memory.store_knowledge("preferred_language", "Python")
    memory.store_knowledge("timezone", "PST")

    # 4. Recall memories
    print("\n4. Memory Recall")
    print("-" * 60)

    results = memory.recall("user", limit=3)
    for i, item in enumerate(results, 1):
        print(f"  {i}. {item.content} (accessed {item.access_count} times)")

    # 5. Get context
    print("\n5. Recent Context")
    print("-" * 60)

    context = memory.get_context(n=5)
    for ctx in context:
        print(f"  [{ctx['type']}] {ctx['content']}")

    # 6. Consolidation
    print("\n6. Memory Consolidation")
    print("-" * 60)

    # Access some memories multiple times to trigger consolidation
    memory.recall("weather", limit=1)
    memory.recall("weather", limit=1)
    memory.recall("weather", limit=1)

    memory.consolidate()

    # 7. Statistics
    print("\n7. Memory Statistics")
    print("-" * 60)

    stats = memory.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 8. Pruning
    print("\n8. Memory Pruning")
    print("-" * 60)

    memory.prune(days=30)

    print("\n✓ Agent Memory System Demo Complete!")


if __name__ == '__main__':
    demo()
